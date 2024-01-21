# Latest
19th Jan, 2024
- Hi guys! Nada and Joseph have consolidated the code into `stereo_odometry.py`. We have included some functions for executing tasks 1-5. The only bottleneck here is that we have not been able to test the code due to a lack of space to download the dataset, so we still need to double-check things.

- If you find any bugs, please fix them and update this note.

- Moses and Joseph will try to test the code tonight, so here's what's left:
  1. Create some nice plots
  2. Attempt the bonus tasks
  3. Evaluate performance

- Plotting and eval
  1. eval/plot_poses.py:
     - Call function plot_trajectory with ground truth file path, estimated file path
     - Creates an interactive 3D plot of the two trajectories
  2. compute_metrics.py
     - Define the base folder. It should contain pose files in format 00_gt.txt and 00_estimated.txt
     - Define the folder where you want to save results
     - runs evaluation on all pose files, plots a lot of stuff and saves the results
     - first for loop for APE, and second for RPE
