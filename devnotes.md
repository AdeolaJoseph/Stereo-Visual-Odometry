# Latest

21st Jan, 2024
- Hi Guys, We've updated the ```stereo_odometry.py``` code. The results are not perfect but you can take a look at them. 
- To run the pipeline you need to have ```dataset/sequences``` folder in the parent dir (```stereo-visual-odometry```). 

- ```config/cfg.yaml``` contains parameters needed for the pipeline to run fine. Please specify the sequence you want to run there.
- After a complete run, the trajectory plot will show up depending on the parameter set in ```cfg.yaml```
- To compare the results with the ground truth, please refer to the the ```__main__``` part of ```plot_poses.py```. 


19th Jan, 2024
- Hi guys! Nada and Joseph have consolidated the code into `stereo_odometry.py`. We have included some functions for executing tasks 1-5. The only bottleneck here is that we have not been able to test the code due to a lack of space to download the dataset, so we still need to double-check things.

- If you find any bugs, please fix them and update this note.

- Moses and Joseph will try to test the code tonight, so here's what's left:
  1. Create some nice plots
  2. Attempt the bonus tasks
  3. Evaluate performance
