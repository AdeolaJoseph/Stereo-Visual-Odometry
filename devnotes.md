# Latest

21st Jan, 2024
- Hi Guys, We've updated the ```stereo_odometry.py``` code. The results are not perfect but you can take a look at them. 
- To run the pipeline you need to have ```dataset/sequences``` folder in the parent dir (```stereo-visual-odometry```). 

- ```config/cfg.yaml``` contains parameters needed for the pipeline to run fine. Please specify the sequence you want to run there.
- After a complete run, the trajectory plot will show up depending on the parameter set in ```cfg.yaml```
- To compare the results with the ground truth, please refer to the the ```__main__``` part of ```plot_poses.py```. 


19th Jan, 2024
- Hi guys! Nada and Joseph have consolidated the code into `stereo_odometry.py`. We have included some functions for executing tasks 1-5. The only bottleneck here is that we have not been able to test the code due to a lack of space to download the dataset, so we still need to double-check things.

22 jan, 2024
- Only bug is that the trajectory is flipped about z. And the trajectory for sequences 04 and 06 are not working.
- Trajectory plotted with GT at end of script



- Plotting and eval
  1. eval/plot_poses.py:
     - Call function plot_trajectory with ground truth file path, estimated file path
     - Creates an interactive 3D plot of the two trajectories
  2. compute_metrics.py
     - Define the base folder. It should contain pose files in format 00_gt.txt and 00_est.txt/00_flip.txt
     - Define the folder where you want to save results
     - runs evaluation on all pose files, plots a lot of stuff and saves the results
     - first for loop for APE, and second for RPE
