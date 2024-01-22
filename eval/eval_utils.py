import numpy as np

def compute_trajectory_length(trajectory_file):
    """
    Compute the length of a trajectory given a KITTI odometry file.

    Parameters:
    trajectory_file (str): The path to the KITTI odometry file.
    
    Returns:
    float: The total length of the trajectory.
    """
    # Load the trajectory file, assuming the file is in the KITTI format with poses as 3x4 transformation matrices
    trajectory = np.loadtxt(trajectory_file)
    
    # Initialize the total length
    total_length = 0.0
    
    # Iterate through the trajectory and compute the incremental distances between consecutive poses
    for i in range(1, len(trajectory)):
        # Previous pose (i-1)
        prev_pose = trajectory[i-1].reshape(3, 4)
        prev_position = prev_pose[:3, 3]
        
        # Current pose (i)
        curr_pose = trajectory[i].reshape(3, 4)
        curr_position = curr_pose[:3, 3]
        
        # Compute the distance between the previous and current position
        distance = np.linalg.norm(curr_position - prev_position)
        
        # Add the distance to the total length
        total_length += distance

    return total_length

def compute_ape_perc(gt_path, ape_rmse):
    leng = compute_trajectory_length(gt_path)
    return round((ape_rmse/leng) * 100.0,2)

ape_rmse = [46.661090,
            1176.512009,
            136.996024,
            8.606559,
            2.772325,
            12.041719,
            3.848314,
            6.181493,
            13.681884,
            18.884405,
            5.948901]

for i in range(11):
    gt_path = f"eval/sim_dataset/{i:02d}_gt.txt"
    ape_rmse_perc = compute_ape_perc(gt_path, ape_rmse[i])
    print(f"{i:02d}: ", ape_rmse_perc)
