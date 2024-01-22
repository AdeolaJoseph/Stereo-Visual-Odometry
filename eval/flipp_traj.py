import numpy as np

def flip_trajectory_along_x(trajectory_file, output_file):
    """
    Flip a KITTI odometry trajectory along the x-axis and save it to a new file.

    Parameters:
    trajectory_file (str): The path to the original KITTI odometry file.
    output_file (str): The path where the flipped trajectory will be saved.
    """
    # Load the trajectory file
    trajectory = np.loadtxt(trajectory_file)

    # Flip the x coordinates (assuming the format is [x, y, z, ...])
    trajectory[:, 11] = -trajectory[:, 11]

    # Save the flipped trajectory to a new file
    np.savetxt(output_file, trajectory, fmt='%1.8e')

for i in range(11):
    flip_trajectory_along_x(trajectory_file=f"eval/results/{i:02d}_est.txt", 
                            output_file=f"eval/results/{i:02d}_flip.txt")