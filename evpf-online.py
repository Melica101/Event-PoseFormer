#!/usr/bin/env python3
import sys
import os
import glob
import copy
import argparse
import datetime
import time

import numpy as np
import yarp

import torch
import torch.nn as nn

# ========== YOUR MOVENET IMPORTS ==========
from pycore.moveenet import MoveNet, Task
from config import cfg
from pycore.moveenet.utils.utils import arg_parser
from pycore.moveenet.task.task_tools import write_output

# ========== POSEFORMER V2 IMPORTS ==========
# Adjust these imports if your module paths differ
from common.model_poseformer import PoseTransformerV2 as Model
from common.camera import *

# ========== UTILS FOR 3D MODEL LOADING ==========

def load_model_weights(model, ckpt_path, device, gpu):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Accept common layouts
    state = (
        ckpt.get("model_pos")
        or ckpt.get("state_dict")
        or ckpt.get("model")
        or ckpt  # raw state_dict
    )

    if not gpu:
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("\nStrict load failed; showing mismatches...")
        model_keys = set(model.state_dict().keys())
        state_keys = set(state.keys())
        missing = sorted(model_keys - state_keys)
        unexpected = sorted(state_keys - model_keys)
        print(f"- Missing ({len(missing)}): {missing[:15]}{' ...' if len(missing) > 15 else ''}")
        print(f"- Unexpected ({len(unexpected)}): {unexpected[:15]}{' ...' if len(unexpected) > 15 else ''}")
        raise e


# ========== MAIN MODULE ==========

class MovenetModule(yarp.RFModule):

    def __init__(self, cfg_in):
        yarp.RFModule.__init__(self)

        # YARP ports
        self.input_port = yarp.BufferedPortImageMono()
        self.output_port = yarp.Port()       # 2D skeleton (13 joints)
        self.output3d_port = yarp.Port()     # 3D skeleton (17 joints)
        self.stamp = yarp.Stamp()

        # Image sizes
        self.image_w = cfg_in['w']
        self.image_h = cfg_in['h']

        # Expected input size to MoveNet
        self.image_w_model = 192
        self.image_h_model = 192

        self.yarp_image = yarp.ImageMono()
        self.yarp_sklt_out = yarp.Bottle()
        self.yarp_sklt3d_out = yarp.Bottle()

        self.checkpoint_path = cfg_in['checkpoint_path']
        self.resultsPath = '/outputs'

        self.fname = None
        self.fname_ts = None
        self.last_timestamp = 0.0
        self.cfg = cfg_in

        # MoveNet 2D
        self.model = None
        self.run_task = None

        # 3D model / PoseFormerV2
        self.model3d = None
        self.device3d = None
        self.args3d = None
        self.kp_buffer = []        # history of 2D joints (17,2)
        self.causal3d = self.cfg.get("causal3d", True)

        self.timing_window = 1.0  # seconds
        self._last_stats_time = time.time()
        self._frame_count = 0
        self._sum_2d = 0.0
        self._sum_3d = 0.0
        self._sum_total = 0.0


    # ---------- 13 → 17 JOINTS (ONLINE VERSION) ----------

    def _convert_13_to_17_single(self, joints13: np.ndarray) -> np.ndarray:
        """
        Convert a single frame of 13-joint 2D pose to 17-joint 2D pose.
        joints13: (13, 2) in image coordinates
        returns joints17: (17, 2), same layout as offline convert_13_to_17_joints()
        """

        mapping_13 = {
            'head': 0, 'shoulder_right': 1, 'shoulder_left': 2,
            'elbow_right': 3, 'elbow_left': 4,
            'hip_left': 5, 'hip_right': 6,
            'wrist_right': 7, 'wrist_left': 8,
            'knee_right': 9, 'knee_left': 10,
            'ankle_right': 11, 'ankle_left': 12
        }

        j = joints13
        kp17 = np.zeros((17, 2), dtype=np.float32)

        # 0–6: pelvis + legs
        kp17[0] = (j[mapping_13['hip_left']] + j[mapping_13['hip_right']]) / 2   # 0: pelvis
        kp17[1] = j[mapping_13['hip_right']]                                     # 1: hip_r
        kp17[2] = j[mapping_13['knee_right']]                                    # 2: knee_r
        kp17[3] = j[mapping_13['ankle_right']]                                   # 3: ankle_r
        kp17[4] = j[mapping_13['hip_left']]                                      # 4: hip_l
        kp17[5] = j[mapping_13['knee_left']]                                     # 5: knee_l
        kp17[6] = j[mapping_13['ankle_left']]                                    # 6: ankle_l

        # neck, spine, nose, head
        neck = (j[mapping_13['shoulder_left']] + j[mapping_13['shoulder_right']]) / 2
        kp17[8] = neck                                                           # 8: neck
        kp17[7] = (kp17[8] + kp17[0]) / 2                                        # 7: spine

        nose = j[mapping_13['head']]
        head = 2 * nose - neck
        kp17[9] = nose                                                           # 9: nose
        kp17[10] = head                                                          # 10: head

        # shoulders / elbows / wrists
        kp17[11] = j[mapping_13['shoulder_left']]                                # 11: LShoulder
        kp17[12] = j[mapping_13['elbow_left']]                                   # 12: LElbow
        kp17[13] = j[mapping_13['wrist_left']]                                   # 13: LWrist

        kp17[14] = j[mapping_13['shoulder_right']]                               # 14: RShoulder
        kp17[15] = j[mapping_13['elbow_right']]                                  # 15: RElbow
        kp17[16] = j[mapping_13['wrist_right']]                                  # 16: RWrist

        return kp17

    # ---------- 3D POSE STEP (LIVE VERSION OF get_pose3D INNER LOOP) ----------

    def _pose3d_step(self, img_h, img_w):
        """
        Uses self.kp_buffer (list of (17,2)) and self.model3d to get 3D pose
        for the current frame (last element in buffer).
        Returns (17,3) or None if not enough data / model not ready.
        """

        if self.model3d is None or self.args3d is None or len(self.kp_buffer) == 0:
            return None

        args = self.args3d
        device = self.device3d

        # Build kp array (T, J, 2) from buffer
        kp = np.stack(self.kp_buffer, axis=0)  # (T,17,2)
        num_frames = kp.shape[0]
        i = num_frames - 1                    # current frame index
        pad = args.pad
        frames = args.frames

        # ---- temporal window selection (causal or non-causal) ----
        if self.causal3d:
            start = max(0, i - pad)
            input_2D_no = kp[start:i+1]  # past + current, (T_window, J, 2)

            if input_2D_no.ndim == 2:
                input_2D_no = input_2D_no[np.newaxis, ...]

            # pad future with current frame
            future_frames = np.repeat(kp[i:i+1], pad, axis=0)
            input_2D_no = np.concatenate((input_2D_no, future_frames), axis=0)
        else:
            start = max(0, i - pad)
            end = min(i + pad, num_frames - 1)
            input_2D_no = kp[start:end+1]  # (T_window,17,2)

            if input_2D_no.ndim == 2:
                input_2D_no = input_2D_no[np.newaxis, ...]

            # pad to exactly frames=243
            left_pad = right_pad = 0
            if input_2D_no.shape[0] != frames:
                if i < pad:
                    left_pad = pad - i
                if i > num_frames - pad - 1:
                    right_pad = i + pad - (num_frames - 1)
                input_2D_no = np.pad(
                    input_2D_no,
                    ((left_pad, right_pad), (0, 0), (0, 0)),
                    mode='edge'
                )

        # ---- normalise (same as offline) ----
        input_2D = normalize_screen_coordinates(
            input_2D_no.copy(), w=img_w, h=img_h
        )

        # ---- flip augmentation ----
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[:, :, 0] *= -1
        input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]

        # Combine normal and flipped
        input_2D = np.concatenate(
            (np.expand_dims(input_2D, axis=0),
             np.expand_dims(input_2D_aug, axis=0)),
            axis=0
        )

        input_2D = input_2D[np.newaxis, :, :, :, :].astype('float32')
        input_2D = torch.from_numpy(input_2D).to(device)

        # ---- 3D inference ----
        with torch.no_grad():
            output_3D_non_flip = self.model3d(input_2D[:, 0])
            output_3D_flip = self.model3d(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

        output_3D = (output_3D_non_flip + output_3D_flip) / 2
        output_3D = output_3D_non_flip
        output_3D[:, :, 0, :] = 0

        post_out = output_3D[0, 0].cpu().detach().numpy()  # (17,3)

        # ---- camera_to_world + normalise depth (same as offline) ----
        rot = np.array(
            [0.1407056450843811,
             -0.1500701755285263,
             -0.755240797996521,
             0.6223280429840088],
            dtype='float32'
        )
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])

        return post_out

    # ---------- RFModule LIFECYCLE ----------

    def configure(self, rf):

        # Initialise YARP
        yarp.Network.init()
        if not yarp.Network.checkNetwork(2):
            print("Could not find network! Run yarpserver and try again.")
            return False

        # set the module name used to name ports
        self.setName((rf.check("name", yarp.Value("/movenet")).asString()))

        # set the output file name
        self.fname = rf.check("write_sk", yarp.Value("pred_2D.npy")).asString()
        self.fname_ts = rf.check("write_ts", yarp.Value("pred_ts.npy")).asString()

        # ---- MoveNet 2D model ----
        self.model = MoveNet(
            num_classes=self.cfg["num_classes"],
            width_mult=self.cfg["width_mult"],
            mode='train'
        )
        self.run_task = Task(self.cfg, self.model)
        self.run_task.modelLoad(self.checkpoint_path)

        # ---- open io ports ----
        if not self.input_port.open(self.getName() + "/img:i"):
            print("Could not open input port")
            return False
        if not self.output_port.open(self.getName() + "/sklt:o"):
            print("Could not open output port")
            return False
        if not self.output3d_port.open(self.getName() + "/sklt3d:o"):
            print("Could not open 3D output port")
            return False

        # ---- PoseFormerV2 3D model ----
        try:
            args3d = argparse.Namespace(
                embed_dim_ratio=32, depth=4,
                frames=243, number_of_kept_frames=27,
                number_of_kept_coeffs=27, pad=(243 - 1) // 2,
                previous_dir='PoseFormerV2-main/checkpoint/',
                n_joints=17, out_joints=17
            )
            self.args3d = args3d

            use_gpu3d = self.cfg.get("gpu3d", True)
            device = torch.device(
                "cuda" if torch.cuda.is_available() and use_gpu3d else "cpu"
            )
            self.device3d = device

            model3d = Model(args=args3d)
            if device.type == "cuda":
                model3d = nn.DataParallel(model3d).to(device)
            else:
                model3d = model3d.to(device)

            ckpt_pattern = os.path.join(args3d.previous_dir, '27_243_45.2.bin')
            model_path = sorted(glob.glob(ckpt_pattern))[0]
            load_model_weights(model3d, model_path, device, gpu=(device.type == "cuda"))
            model3d.eval()
            self.model3d = model3d

            print(f"[3D] PoseFormerV2 loaded from {model_path} on {device}")

        except Exception as e:
            print("[3D] Failed to initialise PoseFormerV2:", e)
            print("[3D] 3D reconstruction will be disabled.")
            self.model3d = None

        return True

    def getPeriod(self):
        # Call updateModule as fast as possible
        return 0.0

    def interruptModule(self):
        self.input_port.interrupt()
        self.output_port.interrupt()
        self.output3d_port.interrupt()
        return True

    def close(self):
        self.input_port.close()
        self.output_port.close()
        self.output3d_port.close()
        return True

    # ---------- MAIN LOOP ----------

    def updateModule(self):

        # Preparing input buffer
        np_input = np.ones((self.image_h, self.image_w), dtype=np.uint8)
        self.yarp_image.resize(self.image_w, self.image_h)
        self.yarp_image.setExternal(np_input.data, np_input.shape[1], np_input.shape[0])

        # Read the image
        read_image = self.input_port.read()
        if read_image is None:
            return False
        self.input_port.getEnvelope(self.stamp)

        stamp_in = self.stamp.getCount() + self.stamp.getTime()

        self.yarp_image.copy(read_image)
        input_image = np.copy(np_input)

        t0 = datetime.datetime.now()

        # # ---- 2D MoveNet prediction ----
        # pre = self.run_task.predict_online(input_image)

        # ----------------- TIMING START -----------------
        t_loop0 = time.perf_counter()

        # 2D inference
        t2d_0 = time.perf_counter()
        pre = self.run_task.predict_online(input_image)
        t2d_1 = time.perf_counter()

        t1 = datetime.datetime.now()
        delta = t1 - t0
        latency = delta.microseconds / 1000.0

        t3d_0 = time.perf_counter()
        # 13-joint 2D (image coordinates)
        joints13 = np.array(pre['joints'], dtype=np.float32).reshape(-1, 2)

        # ---- 17-joint 2D (for 3D model) ----
        joints17_2d = self._convert_13_to_17_single(joints13)

        # Maintain 2D history for 3D temporal model
        self.kp_buffer.append(joints17_2d)
        # keep buffer finite
        if self.args3d is not None and len(self.kp_buffer) > self.args3d.frames + 50:
            self.kp_buffer.pop(0)

        # ---- 3D reconstruction ----
        stamp = yarp.Stamp(0, latency)

        if self.model3d is not None:
            try:
                joints17_3d = self._pose3d_step(self.image_h, self.image_w)
                t3d_1 = time.perf_counter()
            except Exception as e:
                print("[3D] Error in pose3d_step:", e)
                joints17_3d = None

            if joints17_3d is not None:
                self.yarp_sklt3d_out.clear()
                self.yarp_sklt3d_out.addString('SKLT3D')
                lst3d = self.yarp_sklt3d_out.addList()

                flat3d = joints17_3d.flatten()
                for v in flat3d:
                    lst3d.addFloat64(float(v))

                self.output3d_port.setEnvelope(stamp)
                self.output3d_port.write(self.yarp_sklt3d_out)

        # ---- 2D export (unchanged behaviour) ----
        self.last_timestamp = stamp_in

        if self.cfg.get('write_output', False):
            write_output('file.csv', pre['joints'], timestamp=stamp_in)

        self.output_port.setEnvelope(stamp)
        self.yarp_sklt_out.clear()

        out_sklt = np.concatenate((pre['joints'], pre['confidence']))
        self.yarp_sklt_out.addString('SKLT')
        temp_list = self.yarp_sklt_out.addList()
        for v in out_sklt:
            temp_list.addFloat64(float(v))

        self.output_port.write(self.yarp_sklt_out)

        t_loop1 = time.perf_counter()
        # ----------------- TIMING END -----------------

        # per-frame times in ms
        t2d_ms = (t2d_1 - t2d_0) * 1000.0
        t3d_ms = (t3d_1 - t3d_0) * 1000.0
        ttot_ms = (t_loop1 - t_loop0) * 1000.0

        # accumulate
        self._frame_count += 1
        self._sum_2d += t2d_ms
        self._sum_3d += t3d_ms
        self._sum_total += ttot_ms

        now = time.time()
        if now - self._last_stats_time >= self.timing_window:
            # averages over the last window
            n = max(self._frame_count, 1)
            avg_2d = self._sum_2d / n
            avg_3d = self._sum_3d / n
            avg_tot = self._sum_total / n
            hz = 1000.0 / avg_tot if avg_tot > 0 else 0.0

            print(
                f"[{self.getName()}] "
                f"2D: {avg_2d:6.2f} ms | "
                f"3D: {avg_3d:6.2f} ms | "
                f"Total: {avg_tot:6.2f} ms "
                f"({hz:5.1f} Hz)"
            )

            # reset window
            self._frame_count = 0
            self._sum_2d = 0.0
            self._sum_3d = 0.0
            self._sum_total = 0.0
            self._last_stats_time = now

        return True


if __name__ == '__main__':
    # prepare and configure the resource finder
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.setDefaultContext("event-driven")
    rf.configure(sys.argv)

    # parse cfg (same as your original script)
    cfg = arg_parser(cfg)

    # create the module
    module = MovenetModule(cfg)
    module.runModule(rf)