# Stereo Visual Odometry

This project is an implementation of Stereo Visual Odometry (SVO) pipeline using KITTI Dataset. SVO is a method used in computer vision and robotics to estimate the 3D pose (position and orientation) of a camera relative to its starting position, using only the images captured by the camera.

## Features

- Feature detection using SIFT (Scale-Invariant Feature Transform)
- Feature matching using BFMatcher (Brute-Force Matcher)
- 3D-2D motion estimation using PnP (Perspective-n-Point) with RANSAC (Random Sample Consensus)
- Triangulation of points using stereo camera parameters

## Requirements

- Python 3.6 or higher
- OpenCV 4.0 or higher
- NumPy
- Pytorch

## Usage

1. Clone the repository:
``` git clone https://github.com/AdeolaJoseph/Stereo-Visual-Odometry.git```

2. Navigate to the project directory:
```cd Stereo-Visual-Odometry```
3. Run the main script:
```python stereo_odometry.py```

## Authors
1. [Joseph Adeola](https://github.com/AdeolaJoseph)
2. [Khawaja Alamdar](https://github.com/KhAlamdar)
3. [Moses Ebere](https://github.com/MosesEbere)
4. [Nada Abbas](https://github.com/NadaAbbas444)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Results

### 3D Plots

**add a aligned 3d plot with fulcrums here**

### APE vs RPE

The following figure shows the APE error on a simulated drifting trajectory. APE tries to fit the estimated trajectory over the ground truth, and then compute the metrics. This is not ideal as the drift is minimal in beginning and higher in the end and APE is punishing the trajectory in the beginning. It does not compute local trajectory errors. 

<img src="results/sim_results/01_traj.png" alt="testing" height="200">
<img src="results/sim_results/01_APE.png" alt="testing" height="200">

Also, it will be sensitive to outlier poses as it is trying to force the two trajectories to align. Should the odometry be strictly penalized if after a couple of turns it had a few degree error but after that performed flawlessly? [Ivan]

RPE, on the other hand, captures local trajectory errors.

For evaluation with others, its better to use APE as it is standardized.

### Compiled Results:

| Sequence | APE - rmse (m) | APE - rmse (%) |
|----------|----------|----------|
| 00 | 46.71 | 1.25 | 
| 01 | NA | NA |
| 02 | 141.11 | 2.78 |
| 03 | 9.15 | 1.63 |
| 04 | 2.77 | NA |
| 05 | 12.02 | 0.55 |
| 06 | 3.94 | 0.32 |
| 07 | 6.2 | 0.89 |
| 08 | 13.27 | 0.41 |
| 09 | 16.82 | 0.99 |
| 10 | 9.07 | 0.99 |



### 00

- APE 

<img src="results/plots/0p.png" alt="testing" height="200">
<img src="results/plots/0ape2.png" alt="testing" height="200">
<img src="results/plots/0ape1.png" alt="testing" height="200">

- RPE 

<img src="results/plots/0rpe1.png" alt="testing" height="200">
<img src="results/plots/0rpe2.png" alt="testing" height="200">

### 01

- APE 

- RPE 

### 02

- APE 

<img src="results/plots/2p.png" alt="testing" height="200">
<img src="results/plots/2ape2.png" alt="testing" height="200">
<img src="results/plots/2ape1.png" alt="testing" height="200">

- RPE 

<img src="results/plots/2rpe1.png" alt="testing" height="200">
<img src="results/plots/2rpe2.png" alt="testing" height="200">

### 03

- APE 

<img src="results/plots/3p.png" alt="testing" height="200">
<img src="results/plots/3ape2.png" alt="testing" height="200">
<img src="results/plots/3ape1.png" alt="testing" height="200">

- RPE 

<img src="results/plots/3rpe1.png" alt="testing" height="200">
<img src="results/plots/3rpe2.png" alt="testing" height="200">

### 04

- APE 

- RPE 

### 05

- APE 

<img src="results/plots/5p.png" alt="testing" height="200">
<img src="results/plots/5ape2.png" alt="testing" height="200">
<img src="results/plots/5ape1.png" alt="testing" height="200">

- RPE 

<img src="results/plots/5rpe1.png" alt="testing" height="200">
<img src="results/plots/5rpe2.png" alt="testing" height="200">

### 06

- APE 


- RPE 

<img src="results/plots/6rpe1.png" alt="testing" height="200">
<img src="results/plots/6rpe2.png" alt="testing" height="200">

### 07

- APE 

<img src="results/plots/7p.png" alt="testing" height="200">
<img src="results/plots/7ape2.png" alt="testing" height="200">
<img src="results/plots/7ape1.png" alt="testing" height="200">

- RPE 

<img src="results/plots/7rpe1.png" alt="testing" height="200">
<img src="results/plots/7rpe2.png" alt="testing" height="200">

### 08

- APE 

<img src="results/plots/8p.png" alt="testing" height="200">
<img src="results/plots/8ape2.png" alt="testing" height="200">
<img src="results/plots/8ape1.png" alt="testing" height="200">

- RPE 

<img src="results/plots/8rpe1.png" alt="testing" height="200">
<img src="results/plots/8rpe2.png" alt="testing" height="200">

### 09

- APE 

<img src="results/plots/9p.png" alt="testing" height="200">
<img src="results/plots/9ape2.png" alt="testing" height="200">
<img src="results/plots/9ape1.png" alt="testing" height="200">

- RPE 

<img src="results/plots/9rpe1.png" alt="testing" height="200">
<img src="results/plots/9rpe2.png" alt="testing" height="200">

### 10

- APE 

<img src="results/plots/10p.png" alt="testing" height="200">
<img src="results/plots/10ape2.png" alt="testing" height="200">
<img src="results/plots/10ape1.png" alt="testing" height="200">

- RPE 

<img src="results/plots/10rpe1.png" alt="testing" height="200">
<img src="results/plots/10rpe2.png" alt="testing" height="200">

