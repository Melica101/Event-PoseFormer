#!/usr/bin/env python3
import sys
import yarp
import numpy as np
import time


# Use Tk backend
import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class Viewer3DModule(yarp.RFModule):

    def __init__(self):
        yarp.RFModule.__init__(self)

        self.sklt3d_port = yarp.BufferedPortBottle()
        self.stamp = yarp.Stamp()

        self.latest_joints3d = None  # shape (17,3)

        # TK + Matplotlib elements
        self.root = None
        self.fig = None
        self.ax = None
        self.canvas = None

        # Plot handles
        self.joint_scatter = None
        self.bone_lines = []
        self.skeleton_edges = [
            (0, 1), (1, 2), (2, 3),      # right leg
            (0, 4), (4, 5), (5, 6),      # left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # spine + head
            (8, 11), (11, 12), (12, 13),      # left arm
            (8, 14), (14, 15), (15, 16)       # right arm
        ]

        self._fps_t0 = time.time()
        self._fps_counter = 0


    def configure(self, rf):
        # Initialise YARP
        yarp.Network.init()
        if not yarp.Network.checkNetwork(2.0):
            print("Could not find YARP network, run yarpserver.")
            return False

        # Module name
        self.setName(rf.check("name", yarp.Value("/movenetViewer3d")).asString())

        # Open skeleton port
        if not self.sklt3d_port.open(self.getName() + "/sklt3d:i"):
            print("Could not open sklt3d port")
            return False

        # ------------------------------
        # Create Tk window + figure
        # ------------------------------
        self.root = tk.Tk()
        self.root.title("MoveNet 3D Skeleton Viewer")

        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # nice default view
        self.ax.view_init(elev=20, azim=-135)
        self.ax.set_box_aspect([1, 1, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pre-create empty joint scatter
        self.joint_scatter = self.ax.scatter([], [], [], c="k", s=20)

        # Pre-create bone line objects
        self.bone_lines = []
        for _ in self.skeleton_edges:
            line, = self.ax.plot([], [], [], c="r", linewidth=2)
            self.bone_lines.append(line)

        return True


    def getPeriod(self):
        # frame rate
        return 0.01


    def interruptModule(self):
        self.sklt3d_port.interrupt()
        return True


    def close(self):
        self.sklt3d_port.close()
        try:
            self.root.destroy()
        except:
            pass
        return True


    # -----------------------------------------
    # Read /movenet/sklt3d:o
    # -----------------------------------------
    def _update_skeleton3d_from_bottle(self):
        b = self.sklt3d_port.read(False)
        if b is None or b.size() < 2:
            return

        if b.get(0).asString() != "SKLT3D":
            return

        data = b.get(1).asList()
        if data.size() < 17 * 3:
            return

        vals = np.array(
            [data.get(i).asFloat64() for i in range(17 * 3)],
            dtype=np.float32
        )

        J = vals.reshape((17, 3)) * 1000.0
        
        theta = np.radians(-150.0)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
            ], dtype=np.float32)
        
        J_rot = J @ R.T
        
        self.latest_joints3d = J_rot
        # self.latest_joints3d = vals.reshape((17, 3)) * 1000  # to mm


    # -----------------------------------------
    # Update axes contents (NO re-creation)
    # -----------------------------------------
    def _update_plot(self):
        if self.latest_joints3d is None:
            return

        J = self.latest_joints3d

        # center & scale for consistent display
        center = np.mean(J, axis=0)
        Jn = J - center
        max_range = np.max(np.abs(Jn)) + 1e-6

        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        self.ax.set_zlim(-max_range, max_range)

        # update joints
        self.joint_scatter._offsets3d = (Jn[:, 0], Jn[:, 1], Jn[:, 2])

        # update bones
        for line, (i, j) in zip(self.bone_lines, self.skeleton_edges):
            line.set_data([Jn[i, 0], Jn[j, 0]], [Jn[i, 1], Jn[j, 1]])
            line.set_3d_properties([Jn[i, 2], Jn[j, 2]])

        self.canvas.draw_idle()  # non-blocking redraw


    # -----------------------------------------
    # Main loop (YARP periodic thread)
    # -----------------------------------------
    def updateModule(self):
        # Read YARP
        self._update_skeleton3d_from_bottle()

        # Update figure
        self._update_plot()

        # Allow Tk to process mouse/keyboard events
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            # window was closed
            return False
        
        self._fps_counter += 1
        now = time.time()
        if now - self._fps_t0 >= 1.0:
            hz = self._fps_counter / (now - self._fps_t0)
            print(f"[{self.getName()}] Running at {hz:.1f} Hz")
            self._fps_counter = 0
            self._fps_t0 = now

        return True



if __name__ == "__main__":
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.configure(sys.argv)

    module = Viewer3DModule()
    module.runModule(rf)
