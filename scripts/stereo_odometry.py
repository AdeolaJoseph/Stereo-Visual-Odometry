import cv2
import os
import numpy as np
from torch.utils.data import DataLoader
from utils import install_requirements
from dataloader import KITTIDataset
from plotter import Plotter


class StereoOdometry:
    """
    Initializes the StereoOdometry class and provides methods for processing stereo image sequences.

    Args:
        sequence (int): The index of the sequence being processed.
        base_dir (str): The base directory path containing the dataset and results.

    Attributes:
        sequence (int): The sequence index.
        base_dir (str): The base directory path.
        dataset (KITTIDataset): Instance of the KITTI dataset class.
        dataloader (DataLoader): DataLoader object for iterating over the dataset.
        pose (ndarray): The current pose matrix (4x4) of the camera.
        results_dir (str): Directory path for storing results.
        projMatr1 (ndarray): The 3x4 projection matrix for the first camera.
        projMatr2 (ndarray): The 3x4 projection matrix for the second camera.
        colors (dict): Dictionary mapping color names to BGR values.
        camera_matrix (ndarray): The intrinsic camera matrix (3x3).

    Methods:
        calculate_camera_parameters(): Extracts camera parameters from projection matrices.
        load_camera_matrices(sequence): Loads camera projection matrices for a given sequence.
        triangulate_points(matches, keypoints0, keypoints1): Triangulates 3D points from matched keypoints in stereo images.
        compute_pnp(Xk_minus_1, keypoints1, matches): Computes camera pose using the Perspective-n-Point algorithm.
        filter_matches(matches, threshold=30): Filters matches based on a distance threshold.
        project_matches_to_3d(matches, keypoints0, keypoints1, camera_matrix): Projects matched keypoints into 3D space.
        construct_se3_matrix(rotation_vector, translation_vector): Constructs an SE(3) matrix from rotation and translation vectors.
        concatenate_transform(transform1, transform2): Concatenates two SE(3) transformation matrices.
        detect_features(image): Detects SIFT features in an image.
        find_matches(keypoints0, keypoints1, descriptors0, descriptors1): Finds feature matches using BFMatcher.
        save_pose_kitti(file, pose): Saves the camera pose in KITTI format to a file.
        run(p_results): Executes the stereo odometry algorithm on the dataset.
        main(): Entry point for running the class on a dataset sequence.
    """

    def __init__(self, sequence: int, base_dir: str) -> None:
        self.sequence = sequence
        self.base_dir = base_dir
        self.dataset = KITTIDataset(sequence, base_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.pose = np.eye(4, dtype=np.float64)
        self.results_dir = os.path.join(self.base_dir, "dataset", "results")
        self.projMatr1, self.projMatr2 = self.load_projection_matrices(sequence)
        self.colors = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255),}

        self.camera_matrix = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02],
                                       [0.000000e+00, 7.215377e+02, 1.728540e+02],
                                       [0.000000e+00, 0.000000e+00, 1.000000e+00]])

    def calculate_camera_parameters(self):
        """
        Calculates the camera parameters from the projection matrices.

        This method extracts the camera matrix, focal length, principal point, and baseline
        from the projection matrices P0 and P1.

        Returns:
            None
        """
        self.camera_matrix = self.P0[:, :3] 
        #extract the focal length from the projection matrix
        self.f = self.P1[0, 0]
        #extract the principal point from the projection matrix
        self.cu = self.P1[0, 2] 
        self.cv = self.P1[1, 2] 
        #extract the baseline from the projection matrix
        self.base = -self.P1[0, 3] / self.P1[0, 0]
        print(f"focal range: {self.f}")
        print(f"principal point cu: {self.cu}")
        print(f"principal point cv: {self.cv}")
        print(f"baseline: {self.base}")
        
    def load_projection_matrices(self, sequence: int) -> (np.ndarray, np.ndarray):
            """
            Load the projection matrices for a given sequence.

            Args:
                sequence (int): The sequence number.

            Returns:
                tuple: A tuple containing two numpy arrays representing the projection matrices P0 and P1.
            """
            #TODO: We need to check the format of the file containing the projection matrix  
            # Load camera calibration data for the sequence
            calib_file_path = os.path.join(self.base_dir, "dataset", "sequences", f"{sequence:02d}", "calib.txt")
            try:
                with open(calib_file_path, 'r') as file:
                    lines = file.readlines()
                    P0 = np.array([float(value) for value in lines[0].split()[1:13]]).reshape(3, 4)
                    P1 = np.array([float(value) for value in lines[1].split()[1:13]]).reshape(3, 4)
            except Exception:
                print(f"Could not read {calib_file_path}")
            return P0, P1

    def triangulate_points(self, matches: list, keypoints0: list, keypoints1: list) -> np.ndarray:
        """
            Triangulates 3D points from stereo image pairs using matched keypoints.

            Parameters:
            - matches: List of DMatch objects. These are the matches between keypoints
            in two stereo images.
            - keypoints0: List of KeyPoint objects from the first stereo image. 
            - keypoints1: List of KeyPoint objects from the second stereo image.

            Returns:
            - np.ndarray: A 2D array of shape (N, 3), where N is the number of matched
            keypoints. Each row in the array represents the 3D coordinates (X, Y, Z)
            of a point triangulated from the corresponding matched keypoints in the 
            stereo image pair.The function first converts the matched keypoints into 2xN arrays of 2D points (N being
            the number of matches). The function then performs triangulation to get 3D
            points in a homogeneous coordinate system and converts them to a standard
            3D coordinate system.
            """
        # Convert keypoints to the proper format
        points0 = np.float32([keypoints0[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        points1 = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Triangulate points
        points4D = cv2.triangulatePoints(self.projMatr1, self.projMatr2, points0.T, points1.T)
        points3D = cv2.convertPointsFromHomogeneous(points4D.T)
        return points3D.reshape(-1, 3)
    
    def compute_pnp(self, Xk_minus_1: np.ndarray, keypoints1: list, matches: list) -> np.ndarray:
        """
        Computes the Perspective-n-Point (PnP) algorithm to estimate the camera pose.

        Args:
            Xk_minus_1 (np.ndarray): The previous 3D points in the world coordinate system.
            keypoints1 (list): List of keypoints in the first image.
            matches (list): List of matches between keypoints in the first and second image.

        Returns:
            np.ndarray: The estimated camera pose as a 4x4 transformation matrix.
        """
        # Prepare data for solvePnP
        image_points = np.float32([keypoints1[m.trainIdx].pt for m in matches])
        object_points = Xk_minus_1[:len(image_points)]
        # Finds an object pose from 3D-2D point correspondences using the RANSAC scheme
        _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(object_points, image_points, self.projMatr1[:, :3], None)
        return self.construct_se3_matrix(rotation_vector, translation_vector)

    def filter_matches(self, matches: list, threshold: int=30) -> list:
        """
        Filter out matches that have a distance greater than a threshold.
        
        """
        if len(matches) == 0:
            return []
        # Only keep matches with a small Hamming distance
        filtered_matches = [m for m in matches if m.distance < threshold]
        return filtered_matches
    
    def project_matches_to_3d(self, matches: list, keypoints0: list, keypoints1: list, camera_matrix: np.ndarray) -> np.ndarray:
        """
        Project the matches into 3D space using the camera matrix.
        """
        points0 = np.float32([keypoints0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points1 = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        points4D = cv2.triangulatePoints(camera_matrix, camera_matrix, points0, points1)
        points3D = cv2.convertPointsFromHomogeneous(points4D.T)
        return points3D.reshape(-1, 3)

    def construct_se3_matrix(self, rotation_vector: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
        """
        Construct an SE(3) matrix from rotation and translation vectors.
        """
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        se3_matrix = np.eye(4)
        se3_matrix[:3, :3] = rotation_matrix
        se3_matrix[:3, 3] = translation_vector.flatten()
        return se3_matrix

    def concatenate_transform(self, transform1: np.ndarray, transform2: np.ndarray) -> np.ndarray:
        """
        Concatenate two SE(3) transformations.
        """
        return np.dot(transform1, transform2)

    
    def detect_features(self, image: np.ndarray) -> (list, np.ndarray):
        """
        Detects features in the given image using the SIFT algorithm.

        Parameters:
            image (np.ndarray): The input image.

        Returns:
            keypoints (list): List of detected keypoints.
            descriptors (np.ndarray): Array of computed descriptors for the keypoints.
        """
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def find_matches(self, keypoints0: list, keypoints1: list, descriptors0: np.ndarray, descriptors1: np.ndarray) -> list:
        """
        Finds matches between keypoints and descriptors using the BFMatcher algorithm.

        Args:
            keypoints0 (list): List of keypoints from the first image.
            keypoints1 (list): List of keypoints from the second image.
            descriptors0 (ndarray): Descriptors of keypoints from the first image.
            descriptors1 (ndarray): Descriptors of keypoints from the second image.

        Returns:
            list: List of matches between keypoints.
        """
        # Use BFMatcher to match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors0, descriptors1)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def save_pose_kitti(self, file: str, pose: np.ndarray) -> None:
            """
            Saves the pose in the KITTI format to the specified file.

            Args:
                file (file): The file object to write the pose to.
                pose (numpy.ndarray): The pose matrix of shape (4, 4) and dtype np.float64.

            Returns:
                None
            """
            if file and pose.shape == (4, 4) and pose.dtype == np.float64:
                file.write(
                    f"{pose[0,0]} {pose[0,1]} {pose[0,2]} {pose[0,3]} "
                    f"{pose[1,0]} {pose[1,1]} {pose[1,2]} {pose[1,3]} "
                    f"{pose[2,0]} {pose[2,1]} {pose[2,2]} {pose[2,3]}\n")
                    
    def run(self, p_results: str) -> None:
        """
        Runs the stereo odometry algorithm on a sequence of stereo images.

        Args:
            p_results (str): The filepath to save the results.

        Returns:
            None
        """
        pause = False
        for idx, (image0, image1) in enumerate(self.dataloader):
            if image0 is None:
                break
            if image1 is None:
                break    
            #When you load the first  image, you have to save it to get the second set of images to start the feature matching  
            if idx == 0:
                self.save_pose_kitti(p_results, self.pose)
                image0_prev = image0
                image1_prev = image1
                continue

            key = cv2.waitKey(1000000 if pause else 1)
            if key == ord(' '):
                pause = not pause

            # Reset the previous images
            image0_prev = image0
            image1_prev = image1
            
            # TODO: Add your stereo odometry code here
            # The images are now NumPy arrays. You can use them directly in your OpenCV code.
            keypoints0, descriptors0 = self.detect_features(image0_prev)
            keypoints1, descriptors1 = self.detect_features(image0.squeeze(0).numpy())
            matches = self.find_matches(keypoints0, keypoints1, descriptors0, descriptors1)
            filtered_matches=self.filter_matches(self, matches, self.threshold)
            # Ensure we have enough matches
            if len(matches) > 8:  
                Xk_minus_1 = self.triangulate_points(filtered_matches, keypoints0, keypoints1)
                Tk = self.compute_pnp(Xk_minus_1, keypoints1, filtered_matches)

                self.pose = self.concatenate_transform(self.pose, Tk)
                self.save_pose_kitti(p_results, self.pose)
            else:
                print("Not enough matches to compute pose.")

    def main(self):
        with open(os.path.join(self.results_dir, f"{self.sequence:02d}.txt"), "w") as p_results:
            self.run(p_results)

        print(f"Results written to {os.path.join(self.results_dir, f'{self.sequence:02d}.txt')}")
        

if __name__ == "__main__":
    #TODO: Create a config file and place all the parameters needed there
    #TODO: Create a requirements.txt file
    sequence = 9
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pose_estimator = StereoOdometry(sequence, base_dir)
    pose_estimator.run()