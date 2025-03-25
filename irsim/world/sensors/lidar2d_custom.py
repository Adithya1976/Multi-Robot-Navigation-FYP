from calendar import c
import math
from turtle import st
from typing import List

import numpy as np
from shapely import LineString



class CustomLidar2D:
    def __init__(self,
        state=None,
        obj_id=0,
        range_min=0,
        range_max=10,
        angle_range=2*math.pi,
        number=1800,
        external_objects = [],
        **kwargs
    ):
        self.sensor_type = "lidar2d_custom"
        self._state = state
        self.obj_id = obj_id
        self.range_min = range_min
        self.range_max = range_max
        self.angle_range = angle_range
        self.number = number
        self.external_objects = external_objects

        self.angle_min = 0
        self.angle_max = angle_range
        self.angle_inc = angle_range / number
        self.angle_list = np.linspace(self.angle_min, self.angle_max, self.number)

        self.range_data = range_max * np.ones(self.number)

    def plot(self, ax, **kwargs):
        pass

    def plot_clear(self):
        pass

    def step(self, state):
        self._state = state

        # filter objects that are not in the range
        filtered_objects = [
            obj for obj in self.external_objects if self.is_in_range(obj)
        ]

        for obj in filtered_objects:
            self.update_range_data(obj)

    def update_range_data(self, obj):
        if obj.shape == "circle":
            self.update_range_circle(obj)
        elif obj.shape == "linestring":
            self.update_range_linestring(obj)
        else:
            raise NotImplementedError(f"Shape {obj.shape} not implemented")
    
    def update_range_circle(self, obj):
        # find min and max angles w.r.t state
        state = self._state
        obj_state = obj.state
        obj_radius = obj.radius
        obj_angle = math.atan2(obj_state[1, 0] - state[1, 0], obj_state[0, 0] - state[0, 0])
        obj_dist = self.find_dist(state, obj_state)
        half_angle = math.asin(obj_radius / obj_dist)
        obj_angle_min = obj_angle - half_angle - state[2, 0]
        obj_angle_max = obj_angle + half_angle - state[2, 0]

        # adjust angles to be in range 0 to 2pi
        obj_angle_min = obj_angle_min % (2 * math.pi)
        obj_angle_max = obj_angle_max % (2 * math.pi)

        # find the index of the angle
        angle_min_index = math.floor(obj_angle_min / self.angle_inc) + 1
        if angle_min_index == self.number:
            angle_min_index = 0
        angle_max_index = math.floor(obj_angle_max / self.angle_inc)
        if angle_max_index == self.number:
            angle_max_index = 0

        # update range data
        if angle_min_index <= angle_max_index:
            for i in range(angle_min_index, angle_max_index + 1):
                angle = self.angle_list[i] + state[2, 0]
                dist = self.find_intersection_dist_circle(angle, obj_radius, obj_angle, obj_dist)
                if dist < self.range_data[i]:
                    self.range_data[i] = dist
        else:
            for i in range(angle_min_index, self.number):
                angle = self.angle_list[i] + state[2, 0]
                dist = self.find_intersection_dist_circle(angle, obj_radius, obj_angle, obj_dist)
                if dist < self.range_data[i]:
                    self.range_data[i] = dist
            for i in range(0, angle_max_index + 1):
                angle = self.angle_list[i] + state[2, 0]
                dist = self.find_intersection_dist_circle(angle, obj_radius, obj_angle, obj_dist)
                if dist < self.range_data[i]:
                    self.range_data[i] = dist
        
    
    def find_intersection_dist_circle(self, angle, obj_radius, obj_angle, obj_dist):
        theta = abs(angle - obj_angle)

        # x**2 - (2*d*cos(theta)*x) + d**2 - r**2 = 0
        a = 1
        b = -2 * obj_dist * math.cos(theta)
        c = obj_dist ** 2 - obj_radius ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return self.range_max
        x1 = (-b + math.sqrt(discriminant)) / (2 * a)
        x2 = (-b - math.sqrt(discriminant)) / (2 * a)
        dist = min(x1, x2)
        return dist


    def update_range_linestring(self, obj):
        # find min and max angles w.r.t state
        state = self._state
        geometry = obj.geometry
        angle1 = math.atan2(geometry.coords[0][1] - state[1, 0], geometry.coords[0][0] - state[0, 0]) - state[2, 0]
        angle2 = math.atan2(geometry.coords[1][1] - state[1, 0], geometry.coords[1][0] - state[0, 0]) - state[2, 0]

        angle_min = min(angle1, angle2)
        angle_max = max(angle1, angle2)

        # adjust angles to be in range 0 to 2pi
        angle_min = angle_min % (2 * math.pi)
        angle_max = angle_max % (2 * math.pi)

        # find the index of the angle
        angle_min_index = math.floor(angle_min / self.angle_inc) + 1
        if angle_min_index == self.number:
            angle_min_index = 0
        angle_max_index = math.floor(angle_max / self.angle_inc)
        if angle_max_index == self.number:
            angle_max_index = 0

        # update range data
        if angle_min_index <= angle_max_index:
            for i in range(angle_min_index, angle_max_index + 1):
                angle = self.angle_list[i] + state[2, 0]
                dist = self.find_intersection_dist_linestring(state, angle, obj.geometry.coords)
                if dist < self.range_data[i]:
                    self.range_data[i] = dist
        elif angle_min_index - angle_max_index > 1: # to prevent perfectly radial lines from causing errors
            for i in range(angle_min_index, self.number):
                angle = self.angle_list[i] + state[2, 0]
                dist = self.find_intersection_dist_linestring(state, angle, obj.geometry.coords)
                if dist < self.range_data[i]:
                    self.range_data[i] = dist
            for i in range(0, angle_max_index + 1):
                angle = self.angle_list[i] + state[2, 0]
                dist = self.find_intersection_dist_linestring(state, angle, obj.geometry.coords)
                if dist < self.range_data[i]:
                    self.range_data[i] = dist
    
    def find_intersection_dist_linestring(self, state, angle, obj_coords):
        # find the intersection point
        # Compute the direction vectors of the segments
        p0 = (state[0, 0], state[1, 0])
        p1 = (state[0, 0] + self.range_max * math.cos(angle), state[1, 0] + self.range_max * math.sin(angle))
        q0 = obj_coords[0]
        q1 = obj_coords[1]
        r = (p1[0] - p0[0], p1[1] - p0[1])
        s = (q1[0] - q0[0], q1[1] - q0[1])
        
        # Calculate the cross product of r and s
        rxs = r[0] * s[1] - r[1] * s[0]
        
        # If rxs is zero, the lines are parallel (or collinear)
        if rxs == 0:
            return self.range_max

        # Compute the vector from p0 to q0
        qp = (q0[0] - p0[0], q0[1] - p0[1])
        
        # Compute the scalar parameters for the potential intersection point
        t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
        u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
        
        # If t and u are between 0 and 1, the segments intersect
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Intersection point
            # intersection = (p0[0] + t * r[0], p0[1] + t * r[1])
            # Calculate Euclidean distance from p0 to the intersection point
            distance = ((r[0]) ** 2 + (r[1]) ** 2) ** 0.5 * t
            return distance
        else:
            return self.range_max

    
    def is_in_range(self, obj):
        from irsim.world.object_base import ObjectBase
        obj: ObjectBase = obj
        if obj.shape == "circle":
            dist = self.find_dist(self._state, obj.state) - obj.radius
            return dist < self.range_max
        if obj.shape == "linestring":
            geometry: LineString = obj.geometry
            for coord in geometry.coords:
                # convert tuple to np array
                coord = np.array(coord)
                dist = self.find_dist(self._state, np.array(coord))
                if dist < self.range_max:
                    return True
            return False
        raise NotImplementedError(f"Shape {obj.shape} not implemented")

    
    def find_dist(self, state1, state2):
        state1 = state1.flatten()
        state2 = state2.flatten()
        return np.linalg.norm(state1[:2] - state2[:2])
    
    def get_scan(self):
        return {"ranges": self.range_data}