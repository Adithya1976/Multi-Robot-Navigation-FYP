from irsim.world.robots.robot_diff import RobotDiff
import numpy as np
import math
import copy

class VelocityObstacleRobot:
    def __init__(self, robot: RobotDiff, step_time = 0.1):
        self.robot = robot
        self.lidar = self.robot.lidar_custom
        self.lidar_angles = self.lidar.angle_list
        self.lidar_angular_resolution = self.lidar.angle_inc
        self.lidar_number = self.lidar.number
        self.lidar_range_limit = self.lidar.range_max
        self.step_time = step_time
        self.previous_state = copy.deepcopy(robot.state)
        self.current_state = copy.deepcopy(robot.state)
        self.previous_scan = None
        self.current_scan = copy.deepcopy(self.lidar.get_scan()['ranges'])
        self.jump_threshold = 0.2
        self.artificial_obstacle_limit = 4
        self.current_cluster_info = None
        self.previous_cluster_info = None
        self.proprioceptive_observation_dim = 6
        self.exteroceptive_observation_dim = 8
    
    def step(self):
        self.previous_state = copy.deepcopy(self.current_state)
        self.current_state = copy.deepcopy(self.robot.state)

        self.previous_scan = copy.deepcopy(self.current_scan) if self.previous_scan is not None else copy.deepcopy(self.lidar.get_scan()['ranges'])
        self.current_scan = copy.deepcopy(self.lidar.get_scan()['ranges'])
        
    
    def reset(self):
        self.previous_state = copy.deepcopy(self.robot.state)
        self.current_state = copy.deepcopy(self.robot.state)
        self.previous_scan = None
        self.current_scan = copy.deepcopy(self.lidar.get_scan()['ranges'])
    
    def generate_clusters(self, lidar_scan, robot_state):
        labelled_scan = [0] * self.lidar_number
        current_cluster_id = -1

        # Initial clustering of valid readings
        for i, distance in enumerate(lidar_scan):
            # if reading equals (or exceeds) the range limit, mark as -1 (no obstacle)
            if distance > self.lidar_range_limit - 0.01:
                labelled_scan[i] = -1
                continue

            # For the first valid reading or when the previous was invalid, start a new cluster
            if i == 0 or lidar_scan[i-1] > self.lidar_range_limit - 0.01:
                current_cluster_id += 1
                labelled_scan[i] = current_cluster_id
            else:
                # If the previous reading was valid, check the difference.
                prev_distance = lidar_scan[i-1]
                # If the difference is small relative to the previous distance,
                # consider it part of the same obstacle.
                if abs(distance - prev_distance) < self.jump_threshold:
                    labelled_scan[i] = current_cluster_id
                else:
                    # Otherwise, start a new obstacle cluster.
                    current_cluster_id += 1
                    labelled_scan[i] = current_cluster_id

        has_split = False
        # Handle cyclic continuity by merging first and last clusters if appropriate.
        if labelled_scan[0] != -1 and labelled_scan[-1] != -1:
            if abs(lidar_scan[0] - lidar_scan[-1]) < self.jump_threshold:
                i = self.lidar_number - 1
                last_cluster_id = labelled_scan[-1]
                has_split = True
                while i >= 0 and labelled_scan[i] == last_cluster_id:
                    labelled_scan[i] = labelled_scan[0]
                    i -= 1
                current_cluster_id -= 1


        # Create a mapping from cluster id to the list of indices belonging to that cluster.
        clusters = {}
        temp_list = []
        for i, label in enumerate(labelled_scan):
            if label == 0 and i > self.lidar_number/2 and has_split:
                temp_list.append(i)
            elif label != -1:
                clusters.setdefault(label, []).append(i)

        if len(clusters) == 0:
            return {}

        clusters[0] = temp_list + clusters[0]

        cluster_info = {}

        for cluster_id, indices in clusters.items():
            n = len(indices)
            # If the cluster only has one beam, its effective length is 0.
            if n < 2:
                cluster_length = 0.0
            else:
                # For cyclic clusters, the label appears at both start and end.
                # We assume the beams are consecutive on the circle so that the angular span is still (n-1)*0.2.
                theta = (n - 1) * self.lidar_angular_resolution
                # We use the first and last beam distances (order in indices is from low index to high index)
                d1 = lidar_scan[indices[0]]
                d2 = lidar_scan[indices[-1]]
                cluster_length = math.sqrt(d1**2 + d2**2 - 2 * d1 * d2 * math.cos(theta))
            
            # find the average distance of the cluster
            cluster_distance = sum([lidar_scan[i] for i in indices])/len(indices)

            # find minimum distance of the cluster
            cluster_min_distance = min([lidar_scan[i] for i in indices])


            # Filter out the cluster if its length is less than or equal to 0.1.
            if cluster_length > 0.1 and cluster_distance <= self.artificial_obstacle_limit:
                # find cluster centroid
                centroid = self.calculate_cluster_centroid(indices, lidar_scan, robot_state[0][0], robot_state[1][0], robot_state[2][0])

                cluster_info[cluster_id] = {
                    "indices": (indices[0], indices[-1]),
                    "median_index": indices[n//2],
                    "centroid": centroid,
                    "min_distance": cluster_min_distance
                }

        return cluster_info
    
    def calculate_cluster_centroid(self, indices, lidar_scan, robot_x, robot_y, robot_theta):
        sum_x = 0
        sum_y = 0
        for i in indices:
            x = lidar_scan[i] * math.cos(robot_theta + self.lidar_angles[i])
            y = lidar_scan[i] * math.sin(robot_theta + self.lidar_angles[i])
            sum_x += x
            sum_y += y
        return np.array([robot_x + sum_x/len(indices), robot_y + sum_y/len(indices)])
    
    def create_cluster_mapping(self):
        temp = copy.deepcopy(self.current_cluster_info) if self.previous_cluster_info is not None else None
        self.current_cluster_info = self.generate_clusters(self.current_scan, self.current_state)  
        self.previous_cluster_info = temp if temp is not None else copy.deepcopy(self.current_cluster_info)

        current_clusters = self.current_cluster_info
        previous_clusters = self.previous_cluster_info

        cluster_mapping = {}
        # This dict tracks previous cluster assignments: key = previous cluster id,
        # value = (current cluster id, distance)
        assigned_prev = {}

        for curr_id, curr_info in current_clusters.items():
            curr_centroid = curr_info['centroid']
            min_distance = float('inf')
            best_match = None

            # Find the best matching previous cluster for the current cluster
            for prev_id, prev_info in previous_clusters.items():
                prev_centroid = prev_info['centroid']
                distance = np.linalg.norm(curr_centroid - prev_centroid)
                if distance < min_distance:
                    min_distance = distance
                    best_match = prev_id

            if best_match is not None:
                # If this previous cluster isn't assigned, assign it directly.
                if best_match not in assigned_prev:
                    assigned_prev[best_match] = (curr_id, min_distance)
                    cluster_mapping[curr_id] = best_match
                else:
                    # Compare the new distance with the existing one
                    assigned_curr, assigned_distance = assigned_prev[best_match]
                    if min_distance < assigned_distance:
                        # New current cluster is a better match.
                        # Remove the mapping from the previously assigned current cluster.
                        if assigned_curr in cluster_mapping:
                            del cluster_mapping[assigned_curr]
                        # Update the assignment with the new current cluster and distance.
                        assigned_prev[best_match] = (curr_id, min_distance)
                        cluster_mapping[curr_id] = best_match
                    # Otherwise, keep the existing assignment.
        return cluster_mapping
    
    def get_points_from_cluster(self, cluster_info, lidar_scan, robot_state):
        points = []
        start_index, end_index = cluster_info['indices']
        if end_index > start_index:
            for i in range(start_index, end_index + 1):
                x = robot_state[0][0] + lidar_scan[i] * math.cos(robot_state[2][0] + self.lidar_angles[i])
                y = robot_state[1][0] + lidar_scan[i] * math.sin(robot_state[2][0] + self.lidar_angles[i])
                points.append([x, y])
        else:
            for i in range(start_index, self.lidar_number):
                x = robot_state[0][0] + lidar_scan[i] * math.cos(robot_state[2][0] + self.lidar_angles[i])
                y = robot_state[1][0] + lidar_scan[i] * math.sin(robot_state[2][0] + self.lidar_angles[i])
                points.append([x, y])
            for i in range(0, end_index + 1):
                x = robot_state[0][0] + lidar_scan[i] * math.cos(robot_state[2][0] + self.lidar_angles[i])
                y = robot_state[1][0] + lidar_scan[i] * math.sin(robot_state[2][0] + self.lidar_angles[i])
                points.append([x, y])
        points = np.array(points)

        return points
    
    def cal_exp_tim(self, rel_x, rel_y, rel_vx, rel_vy, r):
        # rel_x: xa - xb
        # rel_y: ya - yb

        # (vx2 + vy2)*t2 + (2x*vx + 2*y*vy)*t+x2+y2-(r+mr)2 = 0

        a = rel_vx ** 2 + rel_vy ** 2
        b = 2* rel_x * rel_vx + 2* rel_y * rel_vy
        c = rel_x ** 2 + rel_y ** 2 - r ** 2

        if c <= 0:
            return 0

        temp = b ** 2 - 4 * a * c

        if temp <= 0:
            t = np.inf
        else:
            t1 = ( -b + math.sqrt(temp) ) / (2 * a)
            t2 = ( -b - math.sqrt(temp) ) / (2 * a)

            t3 = t1 if t1 >= 0 else np.inf
            t4 = t2 if t2 >= 0 else np.inf
        
            t = min(t3, t4)

        return t

    def calculate_min_expected_collision_time(self, cluster_info, lidar_scan, cluster_velocity):
        points = self.get_points_from_cluster(cluster_info, lidar_scan, self.robot.state)
        min_t = np.inf
        for i in range(len(points)):
            point = points[i]
            t = self.cal_exp_tim(
                point[0] - self.robot.state[0, 0], 
                point[1] - self.robot.state[1, 0],
                cluster_velocity[0] - self.robot.velocity_xy[0, 0],
                cluster_velocity[1] - self.robot.velocity_xy[1, 0],
                self.robot.radius
            )
            if t < min_t:
                min_t = t
        return min_t
    
    def calculate_desired_velocity(self):
        if self.robot.arrive:
            return np.zeros(2)

        state = self.robot.state[:2].flatten()
        max_velocity = self.robot.vel_max[0][0]
        goal = self.robot.goal[:2].flatten()
        desired_velocity = max_velocity * (goal - state) / np.linalg.norm(goal - state)
        return desired_velocity

    def get_vo_observation(self):
        """
        Returns the VO observation of the robot.
        """
        self.cluster_mapping = self.create_cluster_mapping()
        self.cluster_velocity_mapping = {}

        exteroceptive_observation = []
        minimum_collision_time = np.inf
        
        for cluster_id, prev_cluster_id in self.cluster_mapping.items():

            current_centroid  = self.current_cluster_info[cluster_id]['centroid']
            previous_centroid = self.previous_cluster_info[prev_cluster_id]['centroid']
            x_transalation = current_centroid[0] - previous_centroid[0]
            y_transalation = current_centroid[1] - previous_centroid[1]
            
            cluster_velocity = np.array([x_transalation/self.step_time, y_transalation/self.step_time])
            self.cluster_velocity_mapping[cluster_id] = cluster_velocity

            left_index = self.current_cluster_info[cluster_id]['indices'][1]
            right_index = self.current_cluster_info[cluster_id]['indices'][0]

            half_angle_left = math.asin(self.robot.radius/self.current_scan[left_index])
            half_angle_right = math.asin(self.robot.radius/self.current_scan[right_index])

            left_angle = self.robot.state[2, 0] + self.lidar_angles[left_index] + half_angle_left
            right_angle = self.robot.state[2, 0] + self.lidar_angles[right_index] - half_angle_right

            left_ray = np.array([math.cos(left_angle), math.sin(left_angle)])
            right_ray = np.array([math.cos(right_angle), math.sin(right_angle)])
            
            distance = self.current_cluster_info[cluster_id]['min_distance'] - self.robot.radius
            expected_collision_time = self.calculate_min_expected_collision_time(self.current_cluster_info[cluster_id], self.current_scan, cluster_velocity)
            inverse_expected_collision_time = 1/(expected_collision_time + 0.2)

            obstacle_velocity = cluster_velocity
            obstacle_observation = np.array([obstacle_velocity[0], obstacle_velocity[1], 
                                    left_ray[0], left_ray[1], 
                                    right_ray[0], right_ray[1],
                                    distance, inverse_expected_collision_time], dtype=np.float64)
            exteroceptive_observation.append(obstacle_observation)

            if expected_collision_time < minimum_collision_time:
                minimum_collision_time = expected_collision_time
        
        desired_velocity = self.calculate_desired_velocity()
        proprioceptive_observation = np.array([self.robot.velocity_xy[0][0], self.robot.velocity_xy[1][0],
                                    self.robot.state[2][0], desired_velocity[0],
                                    desired_velocity[1], self.robot.radius], dtype=np.float64)
        
        if len(exteroceptive_observation) == 0:
            exteroceptive_observation = np.zeros((1, 8))
        # convert exterocetive observation to a 2d numpy array
        exteroceptive_observation = np.array(exteroceptive_observation, dtype=np.float64)

        return proprioceptive_observation, exteroceptive_observation, minimum_collision_time