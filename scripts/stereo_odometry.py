import cv2
import os
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import install_requirements, create_dir
from dataloader import KITTIDataset
from plotter import Plotter
from tabulate import tabulate

from typing import Tuple, Dict, List

from plot_poses import plot_trajectory


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

    def __init__(self) -> None:
        self.base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.pose = np.eye(4, dtype=np.float64)
        self.results_dir = os.path.join(self.base_dir, "dataset", "results")
        self.colors = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255),}

        self.camera_matrix = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02],
                                       [0.000000e+00, 7.215377e+02, 1.728540e+02],
                                       [0.000000e+00, 0.000000e+00, 1.000000e+00]])
        
        self.config = self.load_config(os.path.join(self.base_dir, "config", "cfg.yaml"))
        self.threshold = self.config["filter_threshold"]
        self.plotter = Plotter()
        self.show_plots = self.config["show_plots"]
        self.show_tables = self.config["show_tables"]
        self.sanity_check = self.config["sanity_check"]
        self.min_matches = self.config["min_matches"]
        self.translation_history = []
        self.sequence = self.config["sequence"]
        self.dataset = KITTIDataset(self.sequence, self.base_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.projMatr1, self.projMatr2 = self.load_projection_matrices(self.sequence)
        self.intrinsic_params1 = self.calculate_camera_parameters(self.projMatr1)

    # def calculate_camera_parameters(self):
    #     """
    #     Calculates the camera parameters from the projection matrices.

    #     This method extracts the camera matrix, focal length, principal point, and baseline
    #     from the projection matrices P0 and P1.

    #     Returns:
    #         None
    #     """
    #     self.camera_matrix = self.P0[:, :3] 
    #     #extract the focal length from the projection matrix
    #     self.f = self.P1[0, 0]
    #     #extract the principal point from the projection matrix
    #     self.cu = self.P1[0, 2] 
    #     self.cv = self.P1[1, 2] 
    #     #extract the baseline from the projection matrix
    #     self.base = -self.P1[0, 3] / self.P1[0, 0]
    #     print(f"focal range: {self.f}")
    #     print(f"principal point cu: {self.cu}")
    #     print(f"principal point cv: {self.cv}")
    #     print(f"baseline: {self.base}")

    def calculate_camera_parameters(self, P0: np.ndarray):
        """
        Calculates the camera parameters from the projection matrices.

        This method extracts the camera matrix, focal length, principal point, and baseline
        from the projection matrices P0 and P1.

        Returns:
            None
 
        """
        # Extracting the intrinsic camera matrix using RQ decomposition
        # K, R = np.linalg.qr(np.linalg.inv(self.P0[:, :3]))
        # K = np.linalg.inv(K)
        # K /= K[2, 2]  # Normalize the matrix
        # self.focal_lengths = (K[0, 0], K[1, 1])
        # self.principal_point = (K[0, 2], K[1, 2])
        # self.skew = K[0, 1]

        # Use cv2.decomposeProjectionMatrix to extract the rotation and translation matrices
        intrinsic, rotation, translation, _, _, _, _ = cv2.decomposeProjectionMatrix(P0)
        return intrinsic
        
    def load_projection_matrices(self, sequence: int) -> (np.ndarray, np.ndarray):
        """
        Load the projection matrices for a given sequence.

        Args:
            sequence (int): The sequence number.

        Returns:
            tuple: A tuple containing two numpy arrays representing the projection matrices P0 and P1.
        """
        calib_file_path = os.path.join(self.base_dir, "dataset", "sequences", f"{sequence:02d}", "calib.txt")
        try:
            with open(calib_file_path, 'r') as file:
                lines = file.readlines()
                P0 = np.array([float(value) for value in lines[0].split()[1:13]]).reshape(3, 4)
                P1 = np.array([float(value) for value in lines[1].split()[1:13]]).reshape(3, 4)
        except Exception:
            print(f"Could not read {calib_file_path}")
        return P0, P1
    
    def load_config(self, config_file: str) -> dict:
        """
        Loads the configuration file.

        Args:
            config_file (str): The path to the configuration file.

        Returns:
            dict: A dictionary containing the configuration parameters.
        """
        try:
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)
        except Exception:
            print(f"Could not read {config_file}")
        return config

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
        points0 = np.float32([keypoints0[m[0].queryIdx].pt for m in matches]).reshape(-1, 2)
        points1 = np.float32([keypoints1[m[0].trainIdx].pt for m in matches]).reshape(-1, 2)

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
        image_points = np.float32([keypoints1[m[0].trainIdx].pt for m in matches])
        object_points = Xk_minus_1[:len(image_points)]
        # Finds an object pose from 3D-2D point correspondences using the RANSAC scheme
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points, self.intrinsic_params1[:, :3], None, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999, reprojectionError=1)
        # _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(object_points, image_points, self.projMatr1[:, :3], None, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999, reprojectionError=1)
        # _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(object_points, image_points, self.projMatr1[:, :3], None)

        if success:
            # Pose refinement using Levenberg-Marquardt optimization
            # Optional step included to improve results
            rotation_vector, translation_vector = cv2.solvePnPRefineLM(
                object_points[inliers], image_points[inliers], 
                self.intrinsic_params1[:, :3], None, rotation_vector, translation_vector)

        return self.construct_se3_matrix(rotation_vector, translation_vector)
    


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
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptors0, descriptors1, k=2)
        return matches
    
    def filter_matches(self, matches: list, threshold: float=0.75) -> list:
        """
        Filter out matches that have a distance greater than a threshold.

        Args:
            matches (list): List of matches between keypoints in the first and second image.
            threshold (int): Distance threshold for filtering.

        Returns:
            list: Filtered list of matches.
        """
        good_matches = []
        for m, n in matches:
            # Apply Lowe's ratio test
            if m.distance < threshold * n.distance:
                good_matches.append([m])
        return good_matches

    
    def plot_title(self, name: str, sequence: int, idx: Dict[str, int]) -> str:
        """
        Creates a title for the image.

        Args:
            name (str): The name of the image.
            sequence (int): The sequence number.
            idx (dict): A dictionary containing the index of the image with keys representing "left" or "right".

        Returns:
            str: The image title.
        """
        plots = {"Keypoints": f"{name} - Sequence {sequence:02d} - Image {idx}", 
                    "Matches": f"{name} - Sequence {sequence:02d} - Image {idx}",
                    "Trajectory": f"{name} - Sequence {sequence:02d}"
                }
        return plots[name.capitalize()]

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



    def run(self, results_filepath: str) -> None:
        """
        Runs the stereo odometry algorithm on a sequence of stereo images.

        Args:
            results_filepath (str): The filepath to save the odometry results.

        Returns:
            None
        """
        is_paused = False
        for image_index, (left_image, right_image) in enumerate(tqdm(self.dataloader)):
            # Convert the images to NumPy arrays
            left_image = left_image.numpy().squeeze().astype(np.uint8)
            right_image = right_image.numpy().squeeze().astype(np.uint8)

            if left_image is None or right_image is None:
                break

            # Save the first image to start feature matching with the second set
            if image_index == 0:
                self.save_pose_kitti(results_filepath, self.pose)
                previous_left_image = left_image
                previous_right_image = right_image
                continue

            # Control for pausing the process
            key_press = cv2.waitKey(1000000 if is_paused else 1)
            if key_press == ord(' '):
                is_paused = not is_paused

            # Detect features in the previous and current left images
            previous_left_keypoints, previous_left_descriptors = self.detect_features(previous_left_image)
            previous_right_keypoints, previous_right_descriptors = self.detect_features(previous_right_image)
            current_left_keypoints, current_left_descriptors = self.detect_features(left_image)

            # Optional: Draw keypoints
            if self.show_plots:
                self.plotter.draw_keypoints(previous_left_image, previous_left_keypoints, self.plot_title("Keypoints", self.sequence, {"left": f"{image_index-1:06d}"}))
                self.plotter.draw_keypoints(previous_right_image, previous_right_keypoints, self.plot_title("Keypoints", self.sequence, {"right": f"{image_index-1:06d}"}))

            # Find matches between previous and current left images
            left_matches = self.find_matches(previous_left_keypoints, current_left_keypoints, previous_left_descriptors, current_left_descriptors)
            filtered_left_matches = self.filter_matches(left_matches, self.threshold)

            # Optional: Draw matches
            if self.show_plots:
                self.plotter.draw_matches(previous_left_image, left_image, previous_left_keypoints, current_left_keypoints, filtered_left_matches, 
                                self.plot_title("Matches", self.sequence, {"left": f"{image_index-1:06d}", "right": f"{image_index:06d}"}))

            # Prepare data for tabulate
            if self.show_tables:
                table_data = []
                for m in filtered_left_matches:
                    # queryIdx is the index of the feature in previous_left_keypoints
                    # trainIdx is the index of the feature in keypoints0
                    table_data.append([previous_left_keypoints[m[0].queryIdx].pt, current_left_keypoints[m[0].trainIdx].pt])
                print(tabulate(table_data, headers=["Left k-1", "Left k"], tablefmt="fancy_grid"))

            # Keep track of matched keypoints for l,k-1 <-> r,k-1
            previous_matched_left_points = [previous_left_keypoints[match[0].queryIdx].pt for match in filtered_left_matches]
            current_matched_left_points = [current_left_keypoints[match[0].trainIdx].pt for match in filtered_left_matches]

            # Find matches between previous left and right images
            left_to_right_matches = self.find_matches(previous_left_keypoints, previous_right_keypoints, previous_left_descriptors, previous_right_descriptors)
            filtered_left_to_right_matches = self.filter_matches(left_to_right_matches, self.threshold)

            # Keep track of matched keypoints between previous left and right images
            matched_left_points_in_left_to_right = [previous_left_keypoints[match[0].queryIdx].pt for match in filtered_left_to_right_matches]
            matched_right_points_in_left_to_right = [previous_right_keypoints[match[0].trainIdx].pt for match in filtered_left_to_right_matches]


            # Find indices of matched l,k-1 in step 2 in l,k-1 from step 1
            matched_indices = [(i, previous_matched_left_points.index(item)) for i, item in enumerate(matched_left_points_in_left_to_right) if item in previous_matched_left_points]
            
            # indices = [i for i, item in enumerate(list_of_matches1_left_prev) if item in list_of_matches0_left_prev]

            # Use the indices from step 3 to filter out the matched l,k from step 1 - ensure that the feature
            # being considered is in the list of matches from step 1.
            
            filtered_current_left_matches = [filtered_left_matches[i[1]] for i in matched_indices]

            # Triangulate and compute the pose if sufficient matches are found
            if len(filtered_left_to_right_matches) > self.min_matches:
                triangulated_points = self.triangulate_points([filtered_left_to_right_matches[i[0]] for i in matched_indices], 
                                                            previous_left_keypoints, previous_right_keypoints)

                # Project points from step 5 to l,k in step 4
                Tk = self.compute_pnp(triangulated_points, current_left_keypoints, filtered_current_left_matches)

                # Concatenate Tk to the previous pose
                self.pose = self.concatenate_transform(self.pose, Tk)

                # print("Pose: ", self.pose)

                self.save_pose_kitti(results_filepath, self.pose)

                # Plot trajectory
                self.translation_history.append(self.pose[:3, 3].flatten())
                # if self.show_plots:

            else:
                    print("Not enough matches to compute pose.")

            # Update previous images for next iteration
            previous_left_image = left_image
            previous_right_image = right_image

        # Plot the final trajectory
        # self.plot_trajectory(self.translation_history, image_index)

    def main(self):
        # Create the results directory if it does not exist
        create_dir(self.results_dir)
        with open(os.path.join(self.results_dir, f"{self.sequence:02d}_est.txt"), "w") as p_results:
            self.run(p_results)

        print(f"Results written to {os.path.join(self.results_dir, f'{self.sequence:02d}_est.txt')}")
        
        plot_trajectory(os.path.join(self.results_dir, f"{self.sequence:02d}_gt.txt"), 
                        estimated_path = os.path.join(self.results_dir, f"{self.sequence:02d}_est.txt"))

if __name__ == "__main__":
    pose_estimator = StereoOdometry()
    pose_estimator.main()