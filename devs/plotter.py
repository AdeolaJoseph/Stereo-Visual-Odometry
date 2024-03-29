import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plotter:
    """
    A class for plotting visual odometry results.

    Methods:
        draw_matches(image0, image1, matches): Draws the matches between two images.
        draw_keypoints(image, keypoints): Draws the keypoints on the image.
    """
    def __init__(self) -> None:
        ...

    def draw_matches(self, image0: np.ndarray, image1: np.ndarray, kp1: list, kp2: list, 
                     matches: list, plot_title: str = None) -> None:
        """
        Draws the matches between two images.

        Parameters:
            image0 (np.ndarray): The first image.
            image1 (np.ndarray): The second image.
            kp1 (list): List of KeyPoint objects.
            kp2 (list): List of KeyPoint objects.
            matches (list): List of DMatch objects.
            plot_title (str): The title of the plot.

        Returns:
            np.ndarray: The image with the matches drawn on it.
        """
        # Create a new image to draw matches
        result_image = cv2.drawMatches(image0, None, image1, None, matches, None)

        # Display the image with matches
        cv2.imshow('Matches', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_keypoints(self, image: np.ndarray, keypoints: list, plot_title: str = None) -> None:
        """
        Draws the keypoints on the image.

        Parameters:
            image (np.ndarray): The image.
            keypoints (list): List of KeyPoint objects.
            plot_title (str): The title of the plot.

        Returns:
            np.ndarray: The image with the keypoints drawn on it.
        """
        # Draw keypoints on the image
        image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Display the image with keypoints
        cv2.imshow('KeyPoints', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return image
    