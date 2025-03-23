import math
from unittest import result
import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy

from sympy import resultant

def euclidean_distance(point1, point2):
    """
    Euclidean distance between two points.
    :param point1: the first point as a tuple (a_1, a_2, ..., a_n)
    :param point2: the second point as a tuple (b_1, b_2, ..., b_n)
    :return: the Euclidean distance
    """
    a = np.array(point1)
    b = np.array(point2)

    return np.linalg.norm(a - b, ord=2)

def get_resultant_translation(transformation_history):
    overall_transformation = np.eye(3)
    for transformation in transformation_history:
        # Convert the 2x3 transformation matrix into a 3x3 homogeneous transformation matrix.
        T = np.vstack((transformation, np.array([0, 0, 1])))
        # Multiply the transformations (note the order: new transformation on the left)
        overall_transformation = T @ overall_transformation
    resultant_translation_x = overall_transformation[0, 2]
    resultant_translation_y = overall_transformation[1, 2]
    return resultant_translation_x, resultant_translation_y



def point_based_matching(point_pairs):
    """
    This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
    by F. Lu and E. Milios.

    :param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
    :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
    """

    x_mean = 0
    y_mean = 0
    xp_mean = 0
    yp_mean = 0
    n = len(point_pairs)

    if n == 0:
        return None, None, None

    for pair in point_pairs:

        (x, y), (xp, yp) = pair

        x_mean += x
        y_mean += y
        xp_mean += xp
        yp_mean += yp

    x_mean /= n
    y_mean /= n
    xp_mean /= n
    yp_mean /= n

    s_x_xp = 0
    s_y_yp = 0
    s_x_yp = 0
    s_y_xp = 0
    for pair in point_pairs:

        (x, y), (xp, yp) = pair

        s_x_xp += (x - x_mean)*(xp - xp_mean)
        s_y_yp += (y - y_mean)*(yp - yp_mean)
        s_x_yp += (x - x_mean)*(yp - yp_mean)
        s_y_xp += (y - y_mean)*(xp - xp_mean)

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    translation_x = xp_mean - (x_mean*math.cos(rot_angle) - y_mean*math.sin(rot_angle))
    translation_y = yp_mean - (x_mean*math.sin(rot_angle) + y_mean*math.cos(rot_angle))

    return rot_angle, translation_x, translation_y


def icp(reference_points, points, max_iterations=100, distance_threshold=0.3,
        convergence_translation_threshold=1e-3, convergence_rotation_threshold=1e-4,
        point_pairs_threshold=10, verbose=False):
    """
    Modified ICP algorithm that first transforms both the reference points and the points to be aligned
    into a coordinate system with the centroid of the reference points at the origin. The alignment is
    computed in this centered space, and the final transformation is converted back to the original coordinates.
    
    :param reference_points: the reference point set as a numpy array (N x 2)
    :param points: the point set to be aligned as a numpy array (M x 2)
    :param max_iterations: the maximum number of iterations to execute
    :param distance_threshold: the maximum distance to consider two points as a valid pair
    :param convergence_translation_threshold: convergence threshold for the translation parameters (x and y)
    :param convergence_rotation_threshold: convergence threshold for the rotation angle (in rad)
    :param point_pairs_threshold: the minimum number of point pairs required to continue iterating
    :param verbose: whether to print progress messages
    :return: a tuple (total_x_translation, total_y_translation, angle, final_aligned_points, transformed_points)
             where final_aligned_points are the points after ICP (in the original coordinate system) and
             transformed_points are the initial points transformed by the overall transformation.
    """

    transformation_history = []

    # --- Step 1: Compute the centroid of the reference points and center both point sets ---
    ref_centroid = np.mean(reference_points, axis=0)
    reference_points_centered = reference_points - ref_centroid
    points_centered = points - ref_centroid
    initial_points_centered = copy.deepcopy(points_centered)

    # --- Step 2: Initialize the transformation matrix for centered coordinates ---
    resultant_transformation_matrix = np.identity(3)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points_centered)

    # --- Step 3: Run ICP on the centered data ---
    for iter_num in range(max_iterations):
        if verbose:
            print('------ iteration', iter_num, '------')

        closest_point_pairs = []  # list to hold valid point correspondences

        distances, indices = nbrs.kneighbors(points_centered)
        for nn_index in range(len(distances)):
            if distances[nn_index][0] < distance_threshold:
                closest_point_pairs.append((points_centered[nn_index],
                                              reference_points_centered[indices[nn_index][0]]))

        if verbose:
            print('number of pairs found:', len(closest_point_pairs))
        if len(closest_point_pairs) < point_pairs_threshold:
            if verbose:
                print('No better solution can be found (very few point pairs)!')
            break

        # Compute the rotation and translation using the current correspondences
        closest_rot_angle, closest_translation_x, closest_translation_y = point_based_matching(closest_point_pairs)
        if closest_rot_angle is not None:
            if verbose:
                print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                print('Translation:', closest_translation_x, closest_translation_y)
        if (closest_rot_angle is None or
            closest_translation_x is None or
            closest_translation_y is None):
            if verbose:
                print('No better solution can be found!')
            break

        # Construct the rotation matrix and apply the transformation to points_centered
        c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
        rot = np.array([[c, -s],
                        [s,  c]])
        aligned_points_centered = np.dot(points_centered, rot.T)
        aligned_points_centered[:, 0] += closest_translation_x
        aligned_points_centered[:, 1] += closest_translation_y

        # Update the points for the next iteration
        points_centered = aligned_points_centered

        # Update the cumulative transformation matrix in centered coordinates
        rt_matrix = np.hstack((rot, np.array([[closest_translation_x],
                                              [closest_translation_y]])))
        transformation_matrix = np.vstack((rt_matrix, [0, 0, 1]))
        resultant_transformation_matrix = transformation_matrix @ resultant_transformation_matrix

        transformation_history.append(rt_matrix)

        # Check convergence
        if (abs(closest_rot_angle) < convergence_rotation_threshold and
            abs(closest_translation_x) < convergence_translation_threshold and
            abs(closest_translation_y) < convergence_translation_threshold):
            if verbose:
                print('Converged!')
            break

    # --- Step 4: Convert the overall transformation back to the original coordinate system ---
    # The ICP computed transformation is for centered coordinates:
    #   p_transformed_centered = T_centered * (p - ref_centroid)
    # To convert back:
    #   p_transformed = T_centered * (p - ref_centroid) + ref_centroid
    # This is equivalent to an overall transformation with translation:
    T_centered = resultant_transformation_matrix
    # Adjust the translation to account for the original centroid shift
    new_translation = T_centered[0:2, 2] + (ref_centroid - T_centered[0:2, 0:2] @ ref_centroid)
    total_x_translation = new_translation[0]
    total_y_translation = new_translation[1]
    angle = math.atan2(T_centered[1, 0], T_centered[0, 0])

    # Final aligned points in the original coordinate system:
    final_aligned_points = points_centered + ref_centroid

    # Also, compute the transformed version of the initial points (for visualization)
    initial_points_h = np.vstack((initial_points_centered.T, np.ones((1, initial_points_centered.shape[0]))))
    transformed_points_centered = T_centered @ initial_points_h
    transformed_points = (transformed_points_centered[:2, :].T) + ref_centroid

    return total_x_translation, total_y_translation, angle, final_aligned_points, transformed_points
