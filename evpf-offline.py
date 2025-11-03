import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import glob
import copy
import time
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import os
import pathlib
import h5py
import json
import pandas as pd

# Now import the necessary libraries from PoseFormer and hpe-core
sys.path.append(os.getcwd() + '\PoseFormerV2-main')

# PoseFormer imports (originally in PoseFormerV2-main\demo\vis1.py)
from common.model_poseformer import PoseTransformerV2 as Model
from common.camera import *

# hpe-core imports (originally in hpe-core\example\movenet\moveEnet-offline.py)
from tqdm import tqdm
import argparse
import cv2
import json
import pathlib
import os
import numpy as np
import sys, csv
import h5py

sys.path.append(os.getcwd() + '\hpe-core')

from datasets.utils.parsing import import_yarp_skeleton_data, batchIterator
from datasets.utils.events_representation import EROS, eventFrame
from datasets.utils.export import ensure_location, str2bool, get_movenet_keypoints, get_center
from bimvee.importIitYarp import importIitYarp as import_dvs
from bimvee.importIitYarp import importIitYarpBinaryDataLog

from pycore.moveenet import init, MoveNet, Task

from pycore.moveenet.config import cfg
from pycore.moveenet.visualization.visualization import add_skeleton, movenet_to_hpecore
from pycore.moveenet.utils.utils import arg_parser
from pycore.moveenet.task.task_tools import image_show, write_output, superimpose

# Part 1: 2d pose estimation (moveEnet)
def create_ts_list(fps, ts):
    out = dict()
    out['ts'] = list()
    x = np.arange(ts[0], ts[-1], 1 / fps)
    for i in x:
        out['ts'].append(i)
    return out

def import_h5(filename):
    hf = h5py.File(filename, 'r')
    data = np.array(hf["events"][:])
    container = {}
    container['data'] = {}
    container['data']['ch0'] = {}
    container['data']['ch0']['dvs'] = {}
    try:
        container['data']['ch0']['dvs']['ts'] = (data[:, 0]-data[0, 0])*1e-6
    except IndexError:
        print('Error reading .h5 file.')
        exit()
    container['data']['ch0']['dvs']['x'] = data[:, 1]
    container['data']['ch0']['dvs']['y'] = data[:, 2]
    container['data']['ch0']['dvs']['pol'] = data[:, 3].astype(bool)
    return container

def get_representation(rep_name, args):
    if rep_name == 'eros':
        rep = EROS(kernel_size=args.eros_kernel, frame_width=args.frame_width, frame_height=args.frame_height)
    elif rep_name == 'ef':
        rep = eventFrame(frame_height=args.frame_height, frame_width=args.frame_width, n=args.n)
    else:
        print('Representation not found for this setup.')
        exit()
    return rep

def import_file(data_dvs_file):
    filename = os.path.basename(data_dvs_file)
    if filename == 'binaryevents.log':
        data_dvs = importIitYarpBinaryDataLog(filePathOrName=data_dvs_file)
    elif os.path.splitext(filename)[1] == '.h5':
        data_dvs = import_h5(data_dvs_file)
    else:
        data_dvs = import_dvs(filePathOrName=data_dvs_file)
    print('File imported.')
    return data_dvs

def save_event_video_and_csv(data_dvs_file, output_path, args):
    # Initialize the model and task
    init(cfg)
    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    run_task = Task(cfg, model)
    run_task.modelLoad(cfg['ckpt'])

    # Import and process data
    data_dvs = import_file(data_dvs_file)
    channel = list(data_dvs['data'].keys())[0]
    data_dvs['data'][channel]['dvs']['ts'] /= args.ts_scaler
    data_ts = create_ts_list(args.fps, data_dvs['data'][channel]['dvs']['ts'])
    
    iterator = batchIterator(data_dvs['data'][channel]['dvs'], data_ts)
    rep = get_representation(args.rep, args)
    
    if args.write_csv:
        kp_csv = os.path.join(output_path, 'moveEnet_keypoints.csv')
        kp_file = open(kp_csv, 'w', newline='')
        kp_writer = csv.writer(kp_file)
        kp_writer.writerow(['frame', 'joint', 'x', 'y', 'confidence'])

    if args.write_video:
        video_output_path = os.path.join(output_path, "event_video.mp4")
        writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), args.fps,
                             (args.frame_width, args.frame_height))
    
    for fi, (events, pose, batch_size) in enumerate(iterator):
        rep.reset_frame()
        
        if fi % 100 == 0:
            print('frame: ', fi, '/', len(data_ts['ts']))

        if args.stop:
            if fi > args.stop:
                break
            
        for ei in range(batch_size):
            rep.update(vx=int(events['x'][ei]), vy=int(events['y'][ei]))

        if args.skip is not None:
            start, end = map(int, args.skip.split('-'))
            if start <= fi <= end:
                continue

        frame = rep.get_frame()

        if args.rep == 'eros':
            frame = cv2.GaussianBlur(frame, (args.gauss_kernel, args.gauss_kernel), 0)
        if args.write_csv:
            pre = run_task.predict_online(frame, ts=data_ts['ts'][fi])
            output = np.concatenate((pre['joints'].reshape([-1,2]), pre['confidence'].reshape([-1,1])), axis=1)
            for jid, (x, y, c) in enumerate(output):
                kp_writer.writerow([fi, jid, float(x), float(y), float(c)])
        if args.write_video:
           if len(frame.shape) == 2:
              frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

           writer.write(frame)
    
    # Close the CSV file
    if args.write_csv:
        kp_file.close()
        print(f"Keypoints saved to {kp_csv}")

    # Release the video writer if video saving was enabled
    if args.write_video:
        writer.release()
        print(f"Video saved to: {video_output_path}")

# Part 2: Change 2d pose output CSV file to NPZ
def convert_csv_to_npz(csv_path, output_path):
    df = pd.read_csv(csv_path)
    required_cols = {"frame", "joint", "x", "y"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    df["frame"] = df["frame"] - df["frame"].min()

    n_frames = df["frame"].max() + 1
    n_joints = df["joint"].max() + 1

    keypoints = np.zeros((n_frames, n_joints, 2), dtype=np.float32)

    for _, row in df.iterrows():
        f = int(row["frame"])
        j = int(row["joint"])
        keypoints[f, j, 0] = row["x"]
        keypoints[f, j, 1] = row["y"]

    np.savez(output_path, keypoints=keypoints)
    print(f"2d pose csv file converted to npz: {output_path}")

# Part 3: Convert 13-joint file to 17-joint
def convert_13_to_17_joints(input_file, output_file):
    data = np.load(input_file, allow_pickle=True) 
    keypoints_13 = data["keypoints"]

    mapping_13 = {
        'head': 0, 'shoulder_right': 1, 'shoulder_left': 2, 'elbow_right': 3, 'elbow_left': 4,
        'hip_left': 5, 'hip_right': 6, 'wrist_right': 7, 'wrist_left': 8, 'knee_right': 9,
        'knee_left': 10, 'ankle_right': 11, 'ankle_left': 12
    }

    keypoints_17 = np.zeros((keypoints_13.shape[0], 17, 2), dtype=np.float32)
    keypoints_17[:, 0] = (keypoints_13[:, mapping_13['hip_left']] + keypoints_13[:, mapping_13['hip_right']]) / 2
    keypoints_17[:, 1] = keypoints_13[:, mapping_13['hip_right']]
    keypoints_17[:, 2] = keypoints_13[:, mapping_13['knee_right']]
    keypoints_17[:, 3] = keypoints_13[:, mapping_13['ankle_right']]
    keypoints_17[:, 4] = keypoints_13[:, mapping_13['hip_left']]
    keypoints_17[:, 5] = keypoints_13[:, mapping_13['knee_left']]
    keypoints_17[:, 6] = keypoints_13[:, mapping_13['ankle_left']]

    neck = (keypoints_13[:, mapping_13['shoulder_left']] + keypoints_13[:, mapping_13['shoulder_right']]) / 2
    keypoints_17[:, 8] = neck  # Neck

    keypoints_17[:, 7] = (keypoints_17[:, 8] + keypoints_17[:, 0]) / 2  # Spine
    nose = keypoints_13[:, mapping_13['head']]
    head = 2 * nose - neck
    keypoints_17[:, 9] = nose  # Nose
    keypoints_17[:, 10] = head  # Head

    keypoints_17[:, 11] = keypoints_13[:, mapping_13['shoulder_left']]  # LShoulder
    keypoints_17[:, 12] = keypoints_13[:, mapping_13['elbow_left']]  # LElbow
    keypoints_17[:, 13] = keypoints_13[:, mapping_13['wrist_left']]  # LWrist
    keypoints_17[:, 14] = keypoints_13[:, mapping_13['shoulder_right']]  # RShoulder
    keypoints_17[:, 15] = keypoints_13[:, mapping_13['elbow_right']]  # RElbow
    keypoints_17[:, 16] = keypoints_13[:, mapping_13['wrist_right']]  # RWrist

    np.savez(output_file, keypoints=np.array([keypoints_17]))
    print(f"2d pose npz file converted to 17 joints: {output_file}")

# Part 4: 3D Pose Estimation (PoseFormerV2)

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def show2Dpose(kps, img):
    kps = np.asarray(kps)
    if kps.ndim != 2 or kps.shape[1] < 2:
        raise ValueError(f"show2Dpose expects (J,2[,+]), got {kps.shape}")
    kps = kps[:, :2]
    kps_int = np.rint(kps).astype(int)

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, (a, b) in enumerate(connections):
        x1, y1 = kps_int[a]
        x2, y2 = kps_int[b]
        cv2.line(img, (x1, y1), (x2, y2), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (x1, y1), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(img, (x2, y2), radius=3, color=(0, 255, 0), thickness=-1)

    return img
    
def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)

def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path + '/event_video.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5
    print("Extracted FPS:", fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + '/pose/', '*.png')))
    if not names:
        print("No frames found in pose/ directory. Skipping video generation.")
        return
    
    img = cv2.imread(names[0])

    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + f'/3d_demo.mp4', fourcc, fps, size)

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()

def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)
    
def load_model_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Accept common layouts
    state = (
        ckpt.get("model_pos")
        or ckpt.get("state_dict")
        or ckpt.get("model")
        or ckpt  # raw state_dict
    )

    # If keys are like "module.xxx", strip the prefix
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # Try strict load; if it fails, report useful diffs
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("\nStrict load failed; showing mismatches...")
        model_keys = set(model.state_dict().keys())
        state_keys = set(state.keys())
        missing = sorted(model_keys - state_keys)
        unexpected = sorted(state_keys - model_keys)
        print(f"- Missing ({len(missing)}): {missing[:15]}{' ...' if len(missing)>15 else ''}")
        print(f"- Unexpected ({len(unexpected)}): {unexpected[:15]}{' ...' if len(unexpected)>15 else ''}")
        # If you *know* shapes match except for harmless heads, you can relax:
        # model.load_state_dict(state, strict=False)
        raise e
    
def save_predictions_3d(all_predictions, output_dir, filename='predictions_3d.npz'):
    all_predictions = np.concatenate(all_predictions, axis=0)  # Concatenate predictions from all clips
    output_path = os.path.join(output_dir, filename)
    np.savez_compressed(output_path, predictions=all_predictions)
    print(f"All predictions saved to {output_path}")

def get_pose3D(video_path, output_dir, save_images=True, save_demo=True, gpu=False):
    args = argparse.Namespace(
        embed_dim_ratio=32, depth=4, frames=243, number_of_kept_frames=27,
        number_of_kept_coeffs=27, pad=(243 - 1) // 2, previous_dir='PoseformerV2-main/checkpoint/',
        n_joints=17, out_joints=17, gpu=gpu
    )

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = Model(args=args)
    if device.type == "cuda":
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # Load weights
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '27_243_45.2.bin')))[0]
    load_model_weights(model, model_path, device)
    model.eval()

    # Load 2D keypoints
    keypoints = np.load(video_path + '/keypoints_17.npz', allow_pickle=True)['keypoints']
    cap = cv2.VideoCapture(video_path + '/event_video.mp4')
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir_2D = output_dir + '/pose2D/'
    output_dir_3D = output_dir + '/pose3D/'
    os.makedirs(output_dir_2D, exist_ok=True)
    os.makedirs(output_dir_3D, exist_ok=True)

    all_predictions = []
    print("Starting 3D pose estimation...")
    print("Total frames to process:", video_length)
    
    if keypoints.ndim == 4:
      # (P, T, J, 2)
      kp = keypoints[0]
    elif keypoints.ndim == 3:
      # (T, J, 2)
      kp = keypoints
    else:
      raise ValueError(f"Unexpected keypoints shape: {keypoints.shape}")
  
    num_frames = kp.shape[0]
    loop_len = min(video_length, num_frames)

    for i in tqdm(range(loop_len)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape

        start = max(0, i - args.pad)
        end   = min(i + args.pad, num_frames - 1)
        
        # (T_window, J, 2) or possibly (J,2) if start==end
        input_2D_no = kp[start:end+1]
        
        # Ensure 3D (T_window, J, 2)
        if input_2D_no.ndim == 2:
            input_2D_no = input_2D_no[np.newaxis, ...]  # add time axis

        # Pad along time dimension to args.frames
        left_pad = right_pad = 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > num_frames - args.pad - 1:
                right_pad = i + args.pad - (num_frames - 1)
            input_2D_no = np.pad(input_2D_no,((left_pad, right_pad), (0, 0), (0, 0)),mode='edge')

        # Normalize
        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])
        # Flip Augmentation
        joints_left  = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        
        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[:, :, 0] *= -1  # mirror x-coordinates
        input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]

        # Combine normal and flipped
        input_2D = np.concatenate((
            np.expand_dims(input_2D, axis=0),
            np.expand_dims(input_2D_aug, axis=0)
            ), axis=0)

        input_2D = input_2D[np.newaxis, :, :, :, :].astype('float32')
        input_2D = torch.from_numpy(input_2D).to(device)

        with torch.no_grad():
            output_3D_non_flip = model(input_2D[:, 0])
            output_3D_flip = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
        # Average both
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()

        rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])

        if save_images:
            # after padding input_2D_no to (T_window=args.frames, J, 2)
            kps_curr = input_2D_no[args.pad, :, :2]   # shape (J,2), pixels
            image = show2Dpose(kps_curr, copy.deepcopy(img))

            cv2.imwrite(output_dir_2D + str(('%04d'% i)) + '_2D.png', image)

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, ax)
            plt.savefig(output_dir_3D + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
            plt.clf()
            plt.close(fig)

        all_predictions.append(post_out)

    save_predictions_3d(all_predictions, video_path, filename='predictions_3d.npz')
    print('Generating 3D pose successful!')

    if save_demo:
        image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
        image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

        print('\nGenerating demo...')
        for i in tqdm(range(len(image_2d_dir))):
            image_2d = plt.imread(image_2d_dir[i])
            image_3d = plt.imread(image_3d_dir[i])

            ## crop
            edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
            image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

            edge = 130
            image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

            ## show
            font_size = 12
            fig = plt.figure(figsize=(15.0, 5.4))
            ax = plt.subplot(121)
            showimage(ax, image_2d)
            ax.set_title("Input", fontsize = font_size)

            ax = plt.subplot(122)
            showimage(ax, image_3d)
            ax.set_title("Reconstruction", fontsize = font_size)

            ## save
            output_dir_pose = output_dir +'/pose/'
            os.makedirs(output_dir_pose, exist_ok=True)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
            plt.clf()
            plt.close(fig)
            
        img2video(video_path, output_dir)

# Main function to integrate all parts
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_input_path', type=str, required=True, help='Path to the event input folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--write_csv', action='store_true', default=True, help='Write 2D keypoints to CSV')
    parser.add_argument("--write_video", action='store_true', help="Write event video")
    parser.add_argument('--save_images', action='store_true', help='Save 2D and 3D pose images (optional)')
    parser.add_argument('--save_demo', action='store_true', help='Generate demo video (optional)')
    parser.add_argument('--skip', type=str, default=None, help='Skip range of frames')
    parser.add_argument('-eros_kernel', help='EROS kernel size', default=8, type=int)
    parser.add_argument('-frame_width', help='', default=640, type=int)
    parser.add_argument('-frame_height', help='', default=480, type=int)
    parser.add_argument('-gauss_kernel', help='Gaussian filter for EROS', default=7, type=int)
    parser.add_argument('-input', help='Path to input folder (with the data.log file in it)', default=None, type=str)
    parser.add_argument("-ckpt", type=str, default='hpe-core/example/movenet/models/e97_valacc0.81209.pth', help="path to the ckpt. Default: MoveEnet checkpoint.")
    parser.add_argument('-fps', help='Output frame rate', default=50, type=int)
    parser.add_argument('-stop', help='Set to an integer value to stop early after these frames', default=None, type=int)
    parser.add_argument('-rep', help='Representation eros or ef', default='eros', type=str)
    parser.add_argument('-n', help='Number of events in constant count event frame [7500]', default=7500, type=int)
    parser.add_argument("-dev", type=str2bool, nargs='?', const=True, default=False, help="Run in dev mode.")
    parser.add_argument("-ts_scaler", help='', default=1.0, type=float)
    parser.add_argument('-visualise', type=str2bool, nargs='?', default=False, help="Visualise Results [TRUE]")
    args = parser.parse_args()
    
    cfg['ckpt'] = args.ckpt

    # Step 1: Process Event Data into 2D Pose Predictions
    output_dir_current = os.path.join(args.output_dir, f'{os.path.split(os.path.split(args.event_input_path)[0])[1]}')
    os.makedirs(output_dir_current, exist_ok=True)
    save_event_video_and_csv(args.event_input_path, output_dir_current, args)

    # Step 2: Convert CSV to NPZ
    csv_input_path = os.path.join(output_dir_current, "moveEnet_keypoints.csv")
    npz_output_path = os.path.join(output_dir_current, "moveEnet_keypoints.npz")
    convert_csv_to_npz(csv_input_path, npz_output_path)

    # Step 3: Convert 13-joint keypoints to 17-joint keypoints
    keypoints_17_output = os.path.join(output_dir_current, "keypoints_17.npz")
    convert_13_to_17_joints(npz_output_path, keypoints_17_output)

    # Step 4: Generate 3D Pose from the 2D keypoints
    get_pose3D(output_dir_current, output_dir_current, save_images=args.save_images, save_demo=args.save_demo)

if __name__ == '__main__':
    main()