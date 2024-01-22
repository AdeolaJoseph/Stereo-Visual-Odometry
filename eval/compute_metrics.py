import subprocess, os


base_path = "eval/results"
gt_ext = '_gt.txt'
estimated_ext = '_flip.txt'

gt_files = [file for file in os.listdir(base_path) if file.endswith(gt_ext)]
est_files = [file for file in os.listdir(base_path) if file.endswith(estimated_ext)]

print(gt_files)
print(est_files)

for gt_file in gt_files:
    est_file = gt_file.split('_')[0]+estimated_ext
    if est_file in est_files:
        # print(os.path.join(base_path,est_file))
        
        # plot traj
        command = f"evo_traj kitti {os.path.join(base_path,est_file)} {os.path.join(base_path,gt_file)} -p --plot_mode=xz"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
    
        # # plot and save metrics (APE)
        command = f"evo_ape kitti {os.path.join(base_path,est_file)} {os.path.join(base_path,gt_file)} -va --plot --plot_mode xz --r rot_part" #--save_results eval/results/{gt_file.split('_')[0]+'.zip'}"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")

# for RPE
for gt_file in gt_files:
    est_file = gt_file.split('_')[0]+estimated_ext
    if est_file in est_files:
        # plot and save metrics (RPE)
        command = f"evo_rpe kitti {os.path.join(base_path,gt_file)} {os.path.join(base_path,est_file)} --delta 5 --delta_unit f --plot --pose_relation trans_part --plot_mode xz"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")