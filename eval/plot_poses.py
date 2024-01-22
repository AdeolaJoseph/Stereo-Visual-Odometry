import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw_camera_frustum_corrected(ax, position, orientation, scale=1.0, color='k'):
    '''
    position: The position of the camera.
    orientation: The orientation of the camera as a 3x3 rotation matrix.
    scale: Scale of the frustum.
    color: Color of the frustum lines.
    '''

    # Define the camera frustum corners in camera space symmetrically
    # for 0,0
    frustum_corners = np.array([
        [0, 0, 0],  # Camera center
        [-1, 1, 1],  # Top-left corner
        [1, 1, 1],   # Top-right corner
        [1, -1, 1],  # Bottom-right corner
        [-1, -1, 1], # Bottom-left corner
        [-1, 1, 1]   # Closing the loop back to the top-left corner
    ]) * scale

    # Transform the frustum corners to world space
    frustum_corners_world = [position + orientation @ corner for corner in frustum_corners]

    # Draw lines between the camera center and the frustum corners and between the corners
    for i in range(1, len(frustum_corners_world)):
        xs = [frustum_corners_world[0][0], frustum_corners_world[i][0]]
        ys = [frustum_corners_world[0][1], frustum_corners_world[i][1]]
        zs = [frustum_corners_world[0][2], frustum_corners_world[i][2]]
        ax.plot(xs, ys, zs, color)
        if i < len(frustum_corners_world) - 1:  # Draw the base of the frustum
            next_corner = frustum_corners_world[i + 1]
            ax.plot([frustum_corners_world[i][0], next_corner[0]],
                    [frustum_corners_world[i][1], next_corner[1]],
                    [frustum_corners_world[i][2], next_corner[2]], color)

def plot_trajectory(gt_path, estimated_path, draw_viewframes = True, step=200,scale=20):
    '''
    file_path: Path to gt file.
    step: Interval at which to plot the camera frustums.
    scale: Scale of the camera frustums.
    '''
    # c='r'
    files = [gt_path,estimated_path]
    c_traj = ['b','r']
    c_view = ['k','g']
    labels = ['Ground Truth', 'Estimated']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(2):
        # Load and reshape the data to a list of 4x4 matrices
        data = np.loadtxt(files[i]).reshape(-1, 3, 4)
        # Extract the x, y, z coordinates
        x = data[:, 0, 3]
        y = data[:, 1, 3]
        z = data[:, 2, 3]
        if i == 0:
            min_limit = min(min(x), min(y), min(z))
            max_limit = max(max(x), max(y), max(z))
    
        # Plotting
        ax.plot(x, y, z, label=labels[i],c=c_traj[i])
        # Plot camera frustums
        if  draw_viewframes:
            for i in range(0, len(data), step):
                position = data[i, :, 3]
                orientation = data[i, :, :3]
                draw_camera_frustum_corrected(ax, position, orientation,scale)
        
    # Set all three axis limits to the minimum size
    ax.set_xlim(min_limit, max_limit)
    ax.set_ylim(min_limit, max_limit)
    ax.set_zlim(min_limit, max_limit)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


plot_trajectory(gt_path = 'eval/results/00_gt.txt', 
                estimated_path = 'eval/results/00_flip.txt',
                draw_viewframes = True)
