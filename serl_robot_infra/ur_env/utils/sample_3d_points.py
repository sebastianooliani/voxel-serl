import numpy as np
from ur_env.envs.camera_env.config import UR5CameraConfigDualRobot

def shrink_box(min_point, max_point, shrink_factor=0.1):
    """
    Shrink a 3D box by a given factor.
    
    Parameters:
    -----------
    min_point : array-like
        Minimum coordinates of the box
    max_point : array-like
        Maximum coordinates of the box
    shrink_factor : float, optional
        Fraction to shrink the box (default 0.2 or 20%)
    
    Returns:
    --------
    tuple
        (shrunk_min_point, shrunk_max_point)
    """
    # Convert to numpy arrays
    min_point = np.asarray(min_point)
    max_point = np.asarray(max_point)
    
    # Calculate box dimensions
    box_dimensions = max_point - min_point
    
    # Calculate shrinkage
    shrinkage = box_dimensions * shrink_factor
    
    # Shrink the box
    shrunk_min = min_point + shrinkage
    shrunk_max = max_point - shrinkage
    
    return shrunk_min, shrunk_max

def calculate_box_intersection(box1_min, box1_max, box2_min, box2_max):
    """
    Calculate the minimum and maximum points of the intersection between two 3D boxes.
    
    Parameters:
    -----------
    box1_min : array-like
        Minimum coordinates of the first box
    box1_max : array-like
        Maximum coordinates of the first box
    box2_min : array-like
        Minimum coordinates of the second box
    box2_max : array-like
        Maximum coordinates of the second box
    
    Returns:
    --------
    tuple
        (intersection_min, intersection_max) or None if boxes do not intersect
    """
    # Convert to numpy arrays
    box1_min = np.asarray(box1_min)
    box1_max = np.asarray(box1_max)
    box2_min = np.asarray(box2_min)
    box2_max = np.asarray(box2_max)
    
    # Calculate intersection boundaries
    intersection_min = np.maximum(box1_min, box2_min)
    intersection_max = np.minimum(box1_max, box2_max)
    
    # Check if intersection is valid
    if np.any(intersection_max < intersection_min):
        return None
    
    return intersection_min, intersection_max

def sample_points_in_intersecting_boxes(box1_min, box1_max, box2_min, box2_max, num_points, shrink_factor=0.2):
    """
    Sample points uniformly from the intersection of two 3D boxes.
    
    Parameters:
    -----------
    box1_min : array-like
        Minimum coordinates of the first box
    box1_max : array-like
        Maximum coordinates of the first box
    box2_min : array-like
        Minimum coordinates of the second box
    box2_max : array-like
        Maximum coordinates of the second box
    num_points : int
        Number of points to sample
    shrink_factor : float, optional
        Fraction to shrink the boxes (default 0.2 or 20%)
    
    Returns:
    --------
    numpy.ndarray or None
        Array of sampled points, or None if boxes do not intersect
    """
    
    # Calculate intersection of shrunk boxes
    intersection = calculate_box_intersection(
        box1_min, box1_max, 
        box2_min, box2_max
    )
    
    if intersection is None:
        print("The shrunk boxes do not intersect!")
        return None
    
    intersection_min, intersection_max = intersection
    # print("Intersection Min:", intersection_min)
    # print("Intersection Max:", intersection_max)

    # Shrink the intersection box
    box_min_shrunk, box_max_shrunk = shrink_box(intersection_min, intersection_max, shrink_factor)
    # print("Shrunk Intersection Min:", box_min_shrunk)
    # print("Shrunk Intersection Max:", box_max_shrunk)
    
    # Generate points in the intersection box
    random_points = np.random.uniform(0, 1, size=(num_points, 3))
    box_dimensions = box_max_shrunk - box_min_shrunk
    scaled_points = random_points * box_dimensions + box_min_shrunk
    
    return scaled_points

def main():
    config = UR5CameraConfigDualRobot()
    T = config.T_O1_O2
    
    # Example boxes
    box1_min = config.ABS_POSE_LIMIT_LOW_ROBOT_1[:3]
    box1_max = config.ABS_POSE_LIMIT_HIGH_ROBOT_1[:3]
    box2_min = config.ABS_POSE_LIMIT_LOW_ROBOT_2[:3]
    box2_max = config.ABS_POSE_LIMIT_HIGH_ROBOT_2[:3]

    # Evaluate box limits in the correct reference frame
    box2_min = T @ box2_min
    box2_max = T @ box2_max
    
    # Sample points in the intersection
    intersection_points = sample_points_in_intersecting_boxes(
        box1_min, box1_max, box2_min, box2_max, 5
    )
    
    if intersection_points is not None:
        # Verify the points are within the intersection
        print("Sampled Points Shape:", intersection_points.shape)
        print("First 5 Points:")
        print(intersection_points[:5])
        
        # Verify point distribution is within intersection bounds
        print("\nIntersection Points Distribution:")
        print("Min coordinates:", intersection_points.min(axis=0))
        print("Max coordinates:", intersection_points.max(axis=0))

if __name__ == "__main__":
    main()