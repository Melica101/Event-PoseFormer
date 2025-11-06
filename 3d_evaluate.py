import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(pred_path, subject_name, activity_name):
    # === Load predictions ===
    predictions = np.load(pred_path, allow_pickle=True)['predictions']

    # === Load ground truth data ===
    data = np.load("PoseFormerV2-main/data/data_3d_h36m.npz", allow_pickle=True)
    positions_3d = data["positions_3d"].item()

    # Extract subject and activity
    subject = positions_3d[subject_name]
    activity = subject[activity_name]  # shape: (N_frames, 32, 3)
    gt_poses = activity

    # === Select the 17 joints you used ===
    selected_joint_indices = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

    # === Rotation matrix (-135° around Z) ===
    theta = np.radians(-135)
    rot_z_neg_135 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    # === Reshape predictions ===
    if predictions.ndim == 2 and predictions.shape[1] == 3:
        num_frames = predictions.shape[0] // 17
        predictions = predictions.reshape(num_frames, 17, 3)
    else:
        num_frames = predictions.shape[0]

    print("Predictions shape:", predictions.shape)
    print("GT shape:", gt_poses.shape)

    # === Align and compare ===
    num_frames = min(predictions.shape[0], gt_poses.shape[0])
    # gt_poses_selected = gt_poses[:num_frames, selected_joint_indices, :]
    # pred_xyz = predictions[:num_frames]
    
    # Compute start indices so we trim from the beginning of the longer sequence
    start_gt = max(0, gt_poses.shape[0] - num_frames)
    start_pred = max(0, predictions.shape[0] - num_frames)

    gt_poses_selected = gt_poses[start_gt:, selected_joint_indices, :]
    pred_xyz = predictions[start_pred:]

    # Apply rotation
    pred_aligned = (pred_xyz - pred_xyz[:, [0], :]) @ rot_z_neg_135.T
    gt_aligned = gt_poses_selected - gt_poses_selected[:, [0], :]

    # === Compute MPJPE ===
    errors = np.linalg.norm(pred_aligned - gt_aligned, axis=2)  # (frames, 17)
    mpjpe = np.mean(errors) * 1000  # mm
    print(f"MPJPE: {mpjpe:.2f} mm")

    # === Compute PCK@150mm ===
    threshold = 150  # mm
    correct_keypoints = errors * 1000 < threshold
    pck = 100.0 * np.sum(correct_keypoints) / correct_keypoints.size
    print(f"PCK @ {threshold:.0f}mm: {pck:.2f}%")
    
    determine_height(gt_aligned, frame_idx=1000)
    plot_skeleton_3d(pred_aligned, gt_aligned, frame_idx=1000)
    
def plot_skeleton_3d(pred_skel, gt_skel, frame_idx=0):
    """
    Visualize 3D skeletons for a specific frame.
    pred_skel, gt_skel: shape (N_frames, 17, 3)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Select frame
    pred_frame = pred_skel[frame_idx] * 1000  # convert to mm
    gt_frame = gt_skel[frame_idx] * 1000
    
    # Define skeleton edges
    skeleton = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16)
    ]

    # ---- Plot ground truth (blue) ----
    for i, j in skeleton:
        ax.plot(
            [gt_frame[i, 0], gt_frame[j, 0]],
            [gt_frame[i, 1], gt_frame[j, 1]],
            [gt_frame[i, 2], gt_frame[j, 2]],
            color='blue', linewidth=2
        )

    # ---- Plot predicted (red dashed) ----
    for i, j in skeleton:
        ax.plot(
            [pred_frame[i, 0], pred_frame[j, 0]],
            [pred_frame[i, 1], pred_frame[j, 1]],
            [pred_frame[i, 2], pred_frame[j, 2]],
            color='red', linestyle='--', linewidth=2
        )

    # Plot joint points
    gt_scatter = ax.scatter(gt_frame[:, 0], gt_frame[:, 1], gt_frame[:, 2],
                            color='blue', s=30, label='Ground Truth')
    pred_scatter = ax.scatter(pred_frame[:, 0], pred_frame[:, 1], pred_frame[:, 2],
                              color='red', s=30, label='Prediction')

    # Labels and styling
    ax.set_title(f"3D Skeleton Comparison (Frame {frame_idx})", fontsize=13)
    ax.legend(handles=[gt_scatter, pred_scatter])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # Equal aspect ratio
    all_points = np.concatenate([pred_frame, gt_frame], axis=0)
    max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0
    mid = all_points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()

def determine_height(skeleton_data, frame_idx=0):
    """
    Estimate skeleton height from 3D joint data.
    skeleton_data: shape (N_frames, 17, 3)
    """
    # indices
    head_idx = 10
    right_ankle_idx = 3
    left_ankle_idx = 6

    # compute mean ankle height (in m)
    foot_y = np.mean([
        skeleton_data[frame_idx, right_ankle_idx, 2],
        skeleton_data[frame_idx, left_ankle_idx, 2]
        ])
    head_y = skeleton_data[frame_idx, head_idx, 2]

    # height in meters and centimeters
    height_m = abs(head_y - foot_y)
    height_cm = height_m * 100

    print(f"Estimated skeleton height: {height_m:.2f} m ({height_cm:.2f} cm)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D pose predictions against Human3.6M ground truth.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the 3D predictions .npz file.")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g. S1, S5, S11).")
    parser.add_argument("--activity", type=str, required=True, help="Activity name (e.g. Directions, Greeting).")

    args = parser.parse_args()
    main(args.pred_path, args.subject, args.activity)