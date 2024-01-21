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
        # Draw all matches
        img = cv2.drawMatchesKnn(image0, kp1, image1, kp2, matches, None, flags=2)

        # Show the image
        plt.imshow(img)
        if plot_title:
            plt.title(plot_title)
        plt.show()

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

        # Show the image
        plt.imshow(image)
        if plot_title:
            plt.title(plot_title)
        plt.show()

    def plot_trajectory(self, translation_history: list, plot_title: str = None) -> None:
        """
        Plots the trajectory of the camera.

        Parameters:
            translation_history (list): The poses of the camera.
            plot_title (str): The title of the plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # for point in translation_history:
        #     # flatten the point
        #     point = point.flatten()
        #     ax.scatter(point[0], point[1], point[2], c='r', marker='o')

        xs = [point[0] for point in translation_history]
        ys = [point[1] for point in translation_history]
        zs = [point[2] for point in translation_history]

        ax.plot(xs, ys, zs, c='r')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if plot_title:
            plt.title(plot_title)
        plt.show()
    