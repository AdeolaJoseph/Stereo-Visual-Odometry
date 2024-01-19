# Stereo Visual Odometry

This project is an implementation of Stereo Visual Odometry (SVO) in Python using OpenCV. SVO is a method used in computer vision and robotics to estimate the 3D pose (position and orientation) of a camera relative to its starting position, using only the images captured by the camera.

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
