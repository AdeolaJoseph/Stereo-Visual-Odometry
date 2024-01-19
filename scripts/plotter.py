import cv2
import numpy as np


class Plotter:
    """
    A class for plotting visual odometry results.

    Methods:
        draw_matches(image0, image1, matches): Draws the matches between two images.
        draw_keypoints(image, keypoints): Draws the keypoints on the image.
    """
    def __init__(self) -> None:
        ...

    def draw_matches(self, image0: np.ndarray, image1: np.ndarray, matches: list) -> np.ndarray:
        """
        Draws the matches between two images.

        Parameters:
            image0 (np.ndarray): The first image.
            image1 (np.ndarray): The second image.
            matches (list): List of DMatch objects.

        Returns:
            np.ndarray: The image with the matches drawn on it.
        """
        ...

    def draw_keypoints(self, image: np.ndarray, keypoints: list) -> np.ndarray:
        """
        Draws the keypoints on the image.

        Parameters:
            image (np.ndarray): The image.
            keypoints (list): List of KeyPoint objects.

        Returns:
            np.ndarray: The image with the keypoints drawn on it.
        """
        # Draw keypoints on the image
        image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return image
    