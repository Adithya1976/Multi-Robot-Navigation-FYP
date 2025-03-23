from math import e
import time


def line_segment_intersection(p0, p1, q0, q1):
    """
    Find the intersection point between two line segments in 2D.

    Parameters:
        p0, p1: Tuples (x, y) representing the start and end points of the first segment.
        q0, q1: Tuples (x, y) representing the start and end points of the second segment.

    Returns:
        A tuple (x, y) of the intersection point if the segments intersect, or None otherwise.
    """
    # Compute the direction vectors of the segments
    r = (p1[0] - p0[0], p1[1] - p0[1])
    s = (q1[0] - q0[0], q1[1] - q0[1])
    
    # Calculate the cross product of r and s
    rxs = r[0] * s[1] - r[1] * s[0]
    
    # If rxs is zero, the lines are parallel (or collinear)
    if rxs == 0:
        return None

    # Compute the vector from p0 to q0
    qp = (q0[0] - p0[0], q0[1] - p0[1])
    
    # Compute the scalar parameters for the potential intersection point
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
    
    # If t and u are between 0 and 1, the segments intersect
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection = (p0[0] + t * r[0], p0[1] + t * r[1])
        return intersection
    else:
        return None
    
start_time = time.time()
p0 = (-1, -1)
p1 = (1, 1)
q0 = (-1, 1)
q1 = (1, -1)
print(line_segment_intersection(p0, p1, q0, q1))
end_time = time.time()
print("Execution time: ", end_time - start_time)

import numpy as np

def ray_circle_intersection(theta, circle_center, circle_radius):
    """
    Compute the intersection of a ray from the origin (0,0) at angle theta with a circle.
    
    Parameters:
        theta (float): Angle of the ray with respect to the x-axis (in radians).
        circle_center (tuple or array-like): (cx, cy), the center of the circle.
        circle_radius (float): Radius of the circle.
        
    Returns:
        tuple or None: The intersection point (x, y) closest to the origin along the ray,
                       or None if there is no intersection.
    """
    cx, cy = circle_center
    # Compute D = cx*cos(theta) + cy*sin(theta)
    D = cx * np.cos(theta) + cy * np.sin(theta)
    
    # Quadratic coefficients for t: t^2 - 2*D*t + (cx^2 + cy^2 - r^2) = 0
    a = 1.0
    b = -2 * D
    c = cx**2 + cy**2 - circle_radius**2
    
    # Discriminant
    disc = D**2 - c
    
    if disc < 0:
        # No real roots: the ray does not intersect the circle
        return None
    
    sqrt_disc = np.sqrt(disc)
    # Two potential solutions
    t1 = D - sqrt_disc
    t2 = D + sqrt_disc
    
    # We need the smallest nonnegative t (the intersection in front of the origin)
    t_candidates = [t for t in (t1, t2) if t >= 0]
    if not t_candidates:
        return None
    
    t = min(t_candidates)
    # Compute the intersection point using the ray parametric equation
    x = t * np.cos(theta)
    y = t * np.sin(theta)
    
    return (x, y)

# Example usage:
if __name__ == "__main__":
    theta = np.deg2rad(45)  # ray at 45 degrees
    circle_center = (3, 1)  # center of the circle
    circle_radius = 2       # radius of the circle

    start_time = time.time()
    intersection = ray_circle_intersection(theta, circle_center, circle_radius)
    end_time = time.time()
    print("Execution time: ", end_time
          - start_time)
    if intersection is not None:
        print("Intersection point:", intersection)
    else:
        print("No intersection found.")
