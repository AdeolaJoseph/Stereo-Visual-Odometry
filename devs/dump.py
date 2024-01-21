

# # Filter matches between left images at time k-1 and k
# matches_between_left_images = self.filter_matches(left_matches, self.match_threshold)

# # Optional: Draw matches
# if self.show_plots:
#     self.plot_matches(previous_left_image, left_image, previous_left_keypoints, current_left_keypoints, matches_between_left_images, image_index - 1, image_index)

# # Prepare and display match data
# if self.show_tables:
#     self.display_match_table(matches_between_left_images, previous_left_keypoints, current_left_keypoints)

# # Keep track of matched keypoints in left images
# matched_keypoints_in_previous_left = [previous_left_keypoints[match.queryIdx].pt for match in matches_between_left_images]
# matched_keypoints_in_current_left = [current_left_keypoints[match.trainIdx].pt for match in matches_between_left_images]

# # Find matches between previous left and right images
# matches_between_previous_left_and_right = self.find_matches(previous_left_keypoints, previous_right_keypoints, previous_left_descriptors, previous_right_descriptors)
# filtered_matches_between_previous_left_and_right = self.filter_matches(matches_between_previous_left_and_right, self.threshold)

# # Keep track of matched keypoints between previous left and right images
# matched_keypoints_in_previous_left_from_left_right_matches = [previous_left_keypoints[match.queryIdx].pt for match in filtered_matches_between_previous_left_and_right]
# matched_keypoints_in_previous_right_from_left_right_matches = [previous_right_keypoints[match.trainIdx].pt for match in filtered_matches_between_previous_left_and_right]

# # Sanity check
# self.sanity_check_print(len(matched_keypoints_in_previous_left), len(matched_keypoints_in_current_left), len(matched_keypoints_in_previous_left_from_left_right_matches), len(matched_keypoints_in_previous_right_from_left_right_matches))

# # Filter out matched points in current left image based on matches in previous left to right images
# matched_indices = self.find_matched_indices(matched_keypoints_in_previous_left_from_left_right_matches, matched_keypoints_in_previous_left)
# filtered_matches_in_current_left = [matches_between_left_images[i] for i in matched_indices]

# # Triangulate and compute the pose if sufficient matches are found
# if len(filtered_matches_between_previous_left_and_right) > self.minimum_match_count:
#     triangulated_points = self.triangulate_points(filtered_matches_between_previous_left_and_right, previous_left_keypoints, previous_right_keypoints)
#     pose_transform = self.compute_pose_transform(triangulated_points, current_left_keypoints, filtered_matches_in_current_left)
#     self.update_pose(pose_transform)
#     self.save_pose_kitti(results_filepath, self.pose)
#     self.update_translation_history()

# else:
#     print("Not enough matches to compute pose.")

# # Update previous images for next iteration
# previous_left_image = left_image
# previous_right_image = right_image

    # def run(self, p_results: str) -> None:
    #     """
    #     Runs the stereo odometry algorithm on a sequence of stereo images.

    #     Args:
    #         p_results (str): The filepath to save the results.

    #     Returns:
    #         None
    #     """
    #     pause = False
    #     for idx, (image0, image1) in enumerate(tqdm(self.dataloader)):
    #         # Convert the images to NumPy arrays
    #         image0 = image0.numpy().squeeze().astype(np.uint8)
    #         image1 = image1.numpy().squeeze().astype(np.uint8)

    #         if image0 is None:
    #             break
    #         if image1 is None:
    #             break    
    #         #When you load the first  image, you have to save it to get the second set of images to start the feature matching  
    #         if idx == 0:
    #             self.save_pose_kitti(p_results, self.pose)
    #             image0_prev = image0
    #             image1_prev = image1
    #             continue

    #         key = cv2.waitKey(1000000 if pause else 1)
    #         if key == ord(' '):
    #             pause = not pause
            
    #         # TODO: Add your stereo odometry code here
    #         # The images are now NumPy arrays. You can use them directly in your OpenCV code.
    #         """
    #         keypoints0_prev = left keypoints in image k-1
    #         keypoints1_prev = right keypoints in image k-1
    #         keypoints0 = left keypoints in image k
    #         keypoints1 = right keypoints in image k
    #         descriptors0_prev = left descriptors in image k-1
    #         descriptors1_prev = right descriptors in image k-1
    #         descriptors0 = left descriptors in image k
    #         descriptors1 = right descriptors in image k

    #         matches0 = matches between keypoints0_prev (previous left) and keypoints0 (current left)
    #         matches1 = matches between keypoints0_prev (previous left) and keypoints1_prev (previous right)

    #         filtered_matches0 = matches0 filtered by threshold
    #         filtered_matches1 = matches1 filtered by threshold

    #         list_of_matches0_left_prev = list of left keypoints in image k-1 from filtered_matches0
    #         list_of_matches0_left_current = list of left keypoints in image k from filtered_matches0
    #         list_of_matches1_left_prev = list of left keypoints in image k-1 from filtered_matches1
    #         list_of_matches1_right_prev = list of right keypoints in image k-1 from filtered_matches1

    #         indices = (i, j) where i is the index of the feature in list_of_matches1_left_prev and j is the index of the feature in list_of_matches0_left_prev
    #         filtered_matches2 = filtered_matches0 filtered by indices
    #         """
                
    #         # Detect features l,k-1; l,k; r,k-1
    #         keypoints0_prev, descriptors0_prev = self.detect_features(image0_prev)
    #         keypoints1_prev, descriptors1_prev = self.detect_features(image1_prev)
    #         keypoints0, descriptors0 = self.detect_features(image0)

    #         # Draw keypoints
    #         if self.show_plots:
    #             self.plotter.draw_keypoints(image0_prev, keypoints0_prev, self.plot_title("Keypoints", self.sequence, {"left": f"{idx-1:06d}"}))
    #             self.plotter.draw_keypoints(image1_prev, keypoints1_prev, self.plot_title("Keypoints", self.sequence, {"right": f"{idx-1:06d}"}))

    #         # Find matches l,k-1 <-> l,k
    #         matches0 = self.find_matches(keypoints0_prev, keypoints0, descriptors0_prev, descriptors0)

    #         # Draw matches
    #         filtered_matches0 = self.filter_matches(matches0, self.threshold)
    #         if self.show_plots:
    #             self.plotter.draw_matches(image0_prev, image0, keypoints0_prev, keypoints0, filtered_matches0, 
    #                                     self.plot_title("Matches", self.sequence, {"left": f"{idx-1:06d}", "right": f"{idx:06d}"}))

    #         # Prepare data for tabulate
    #         if self.show_tables:
    #             table_data = []
    #             for m in filtered_matches0:
    #                 # queryIdx is the index of the feature in keypoints0_prev
    #                 # trainIdx is the index of the feature in keypoints0
    #                 table_data.append([keypoints0_prev[m[0].queryIdx].pt, keypoints0[m[0].trainIdx].pt])
    #             print(tabulate(table_data, headers=["Left k-1", "Left k"], tablefmt="fancy_grid"))

    #         # Keep history of matches for l,k-1 <-> l,k
    #         list_of_matches0_left_prev = [keypoints0_prev[m[0].queryIdx].pt for m in filtered_matches0]
    #         list_of_matches0_left_current = [keypoints0[m[0].trainIdx].pt for m in filtered_matches0]


    #         # TODO:
    #         # 1. Find matches l,k-1 <-> l,k - DONE
    #         # 2. Find matches l,k-1 <-> r,k-1
    #         # 3. Find indices of matched l,k-1 in step 2 in l,k-1 from step 1
    #         # 4. Use the indices from step 3 to filter out the matched l,k from step 1
    #         # 5. Triangulate points from filtered matches from step 2
    #         # 6. Project points from step 5 to l,k in step 4 and compute PnP to get the pose Tk
    #         # 7. Concatenate Tk to the previous pose
                
    #         # Find matches l,k-1 <-> r,k-1
    #         matches1 = self.find_matches(keypoints0_prev, keypoints1_prev, descriptors0_prev, descriptors1_prev)
    #         filtered_matches1 = self.filter_matches(matches1, self.threshold)

    #         # Keep history of matches for l,k-1 <-> r,k-1
    #         list_of_matches1_left_prev = [keypoints0_prev[m[0].queryIdx].pt for m in filtered_matches1]
    #         list_of_matches1_right_prev = [keypoints1_prev[m[0].trainIdx].pt for m in filtered_matches1]

    #         # if self.sanity_check:
    #         #     print("Len list_of_matches0_left_prev: ", len(list_of_matches0_left_prev))
    #         #     print("Len list_of_matches0_left_current: ", len(list_of_matches0_left_current))
    #         #     print("Len list_of_matches1_left_prev: ", len(list_of_matches1_left_prev))
    #         #     print("Len list_of_matches1_right_prev: ", len(list_of_matches1_right_prev))

    #         # Find indices of matched l,k-1 in step 2 in l,k-1 from step 1
    #         indices = [(i, list_of_matches0_left_prev.index(item)) for i, item in enumerate(list_of_matches1_left_prev) if item in list_of_matches0_left_prev]
    #         # indices = [i for i, item in enumerate(list_of_matches1_left_prev) if item in list_of_matches0_left_prev]

    #         # Use the indices from step 3 to filter out the matched l,k from step 1 - ensure that the feature
    #         # being considered is in the list of matches from step 1.
    #         filtered_matches2 = [filtered_matches0[i[1]] for i in indices]
    #         # filtered_matches2 = [filtered_matches0[i] for i in indices if i < len(filtered_matches0)]

    #         if self.sanity_check:
    #             a = [keypoints0_prev[m[0].queryIdx].pt for m in filtered_matches2]    
    #             # b = [keypoints0[m[0].trainIdx].pt for m in filtered_matches2] 
    #             # print(a[50], list_of_matches0_left_prev[int(indices[50])])       

    #         # Triangulate points from filtered matches from step 2
    #         if len(filtered_matches1) > self.min_matches:
    #             Xk_minus_1 = self.triangulate_points([filtered_matches1[i[0]] for i in indices], keypoints0_prev, keypoints1_prev)

    #             # Project points from step 5 to l,k in step 4
    #             Tk = self.compute_pnp(Xk_minus_1, keypoints0, filtered_matches2)

    #             # Concatenate Tk to the previous pose
    #             self.pose = self.concatenate_transform(self.pose, Tk)

    #             # print("Pose: ", self.pose)

    #             self.save_pose_kitti(p_results, self.pose)

    #             # Plot trajectory
    #             self.translation_history.append(self.pose[:3, 3].flatten())
    #             # if self.show_plots:

    #         else:
    #             print("Not enough matches to compute pose.")

    #         # Reset the previous images
    #         image0_prev = image0
    #         image1_prev = image1


    #     self.plotter.plot_trajectory(self.translation_history, self.plot_title("Trajectory", self.sequence, {"left": f"{idx:06d}"}))
