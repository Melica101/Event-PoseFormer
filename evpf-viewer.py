#!/usr/bin/env python3
import sys
import yarp
import numpy as np
import cv2


class Viewer3DModule(yarp.RFModule):
    def __init__(self):
        yarp.RFModule.__init__(self)

        # Input: 3D skeleton bottle
        self.sklt3d_port = yarp.BufferedPortBottle()
        self.stamp = yarp.Stamp()

        # Output: RGB image for yarpview
        self.out_port = yarp.BufferedPortImageRgb()

        self.latest_joints3d = None  # (17, 3)
        self.angle = -135.0            # initial rotation angle around Y
        self.spin_speed = 0.5        # degrees per update

        self.canvas_h = 400
        self.canvas_w = 400

    def configure(self, rf):
        # Initialise YARP
        yarp.Network.init()
        if not yarp.Network.checkNetwork(2.0):
            print("Could not find YARP network, run yarpserver.")
            return False

        # Module name
        self.setName(rf.check("name", yarp.Value("/evpfViewer")).asString())

        # Open 3D input port
        if not self.sklt3d_port.open(self.getName() + "/sklt3d:i"):
            print("Could not open 3D skeleton input port")
            return False

        # Open RGB output port
        if not self.out_port.open(self.getName() + "/3d:o"):
            print("Could not open 3D image output port")
            return False

        return True

    def getPeriod(self):
        # ~50 Hz
        return 0.01

    def interruptModule(self):
        self.sklt3d_port.interrupt()
        self.out_port.interrupt()
        return True

    def close(self):
        self.sklt3d_port.close()
        self.out_port.close()
        return True

    # --- Helper: parse /movenet/sklt3d:o Bottle (17 joints, 3D) ---
    def _update_skeleton3d_from_bottle(self):
        b = self.sklt3d_port.read(False)  # non-blocking
        if b is None:
            return

        if b.size() < 2:
            return

        tag = b.get(0).asString()
        if tag != "SKLT3D":
            return

        data_list = b.get(1).asList()
        n_vals = data_list.size()

        # Expect at least 17*3 = 51 floats
        if n_vals < 17 * 3:
            return

        vals = np.array(
            [data_list.get(i).asFloat64() for i in range(17 * 3)],
            dtype=np.float32
        )

        joints3d = vals.reshape((17, 3))
        self.latest_joints3d = joints3d

    # --- Simple 3D→2D projection with rotation around Y-axis ---
    def _project_points_3d_to_2d(self, joints3d, angle_deg=45.0,
                                 scale=120, offset=(200, 220)):
        """
        joints3d: (17, 3)
        Returns 2D projected joints: (17, 2) in image pixels.

        We remap axes because the 3D model uses:
        X = left/right
        Y = depth
        Z = up/down (height)
        and we want:
        X' = left/right
        Y' = up/down
        Z' = depth
        """

        # --- axis remap: [X, Z, Y] so that Z becomes vertical ---
        X = joints3d[:, 0]
        Z = joints3d[:, 2]   # height
        Y = joints3d[:, 1]   # depth

        J = np.stack([X, Z, Y], axis=1).astype(np.float32)  # (17,3)
        # Now:
        # J[:,0] = left/right
        # J[:,1] = up/down
        # J[:,2] = depth

        # --- rotate around the vertical axis (Y=up) or depth, pick what looks nicer ---
        theta = np.radians(angle_deg)

        # rotate around vertical axis (Y=up) -> spin around the person
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0,             1,             0],
            [-np.sin(theta), 0, np.cos(theta)],
        ], dtype=np.float32)

        J_rot = J @ R.T  # (17,3)

        # --- simple orthographic projection ---
        u = J_rot[:, 0] * scale + offset[0]      # left/right
        v = -J_rot[:, 1] * scale + offset[1]     # up/down (invert for image)

        return np.vstack([u, v]).T.astype(int)


    def _draw_skeleton3d_cv(self, canvas, joints2d):
        """
        canvas: (H,W,3) uint8
        joints2d: (17,2) int pixel coords
        """

        # 17-joint mapping we used in the 3D pipeline:
        # 0: pelvis
        # 1: hip_r
        # 2: knee_r
        # 3: ankle_r
        # 4: hip_l
        # 5: knee_l
        # 6: ankle_l
        # 7: spine
        # 8: neck
        # 9: nose
        # 10: head
        # 11: LShoulder
        # 12: LElbow
        # 13: LWrist
        # 14: RShoulder
        # 15: RElbow
        # 16: RWrist

        skeleton_edges = [
            (0, 1), (1, 2), (2, 3),      # right leg
            (0, 4), (4, 5), (5, 6),      # left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # spine + head
            (8, 11), (11, 12), (12, 13),      # left arm
            (8, 14), (14, 15), (15, 16)       # right arm
        ]

        # Draw bones
        for i, j in skeleton_edges:
            p1 = tuple(joints2d[i])
            p2 = tuple(joints2d[j])
            cv2.line(canvas, p1, p2, (0, 0, 255), 2)

        # Draw joints
        for x, y in joints2d:
            cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 0), -1)

    # --- main periodic loop ---
    def updateModule(self):
        # update latest 3D skeleton (if new message arrived)
        self._update_skeleton3d_from_bottle()

        # prepare canvas (black)
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        if self.latest_joints3d is not None:
            # slowly rotate camera
            self.angle += self.spin_speed
            if self.angle >= 360.0:
                self.angle -= 360.0

            joints2d = self._project_points_3d_to_2d(
                self.latest_joints3d,
                angle_deg=self.angle,
                scale=120,
                offset=(self.canvas_w // 2, int(self.canvas_h * 0.65)),
            )
            self._draw_skeleton3d_cv(canvas, joints2d)

        # publish canvas as RGB image
        out_img = self.out_port.prepare()
        out_img.resize(self.canvas_w, self.canvas_h)
        # attach numpy buffer
        out_img.setExternal(canvas.data, self.canvas_w, self.canvas_h)
        self.out_port.write()

        return True


if __name__ == "__main__":
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.setDefaultContext("event-driven")
    rf.configure(sys.argv)

    module = Viewer3DModule()
    module.runModule(rf)