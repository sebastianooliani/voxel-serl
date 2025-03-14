import numpy as np
from ur_env.envs.camera_env.config import UR5CameraConfigDualRobot
import matplotlib.pyplot as plt

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
    # Ensure min and max are correctly ordered
    box1_min_temp = np.minimum(box1_min, box1_max)
    box1_max_temp = np.maximum(box1_min, box1_max)
    box2_min_temp = np.minimum(box2_min, box2_max)
    box2_max_temp = np.maximum(box2_min, box2_max)

    box1_min = box1_min_temp
    box1_max = box1_max_temp
    box2_min = box2_min_temp
    box2_max = box2_max_temp

    # Calculate intersection boundaries
    intersection_min = np.maximum(box1_min, box2_min)
    intersection_max = np.minimum(box1_max, box2_max)
    
    # More robust intersection check
    if np.all(intersection_max >= intersection_min - 1e-10):
        return intersection_min, intersection_max
    
    return None

def sample_points_in_intersecting_boxes(
        box1_min=np.array([-0.605, -0.533, 0.128]), 
        box1_max=np.array([-0.263, -0.363, 0.377]),
        box2_min=np.array([-0.650, -0.569, 0.114]),
        box2_max=np.array([-0.352, 0.369, 0.387]),
        num_points=1, 
        shrink_factor=0.2, 
        seed=42):
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
        print("The boxes do not intersect!")
        return None
    
    intersection_min, intersection_max = intersection

    # subtract 0.13 from the max z value 
    intersection_max[0] -= 0.07
    intersection_max[1] -= 0.07
    intersection_max[2] -= 0.13
    intersection_min[0] += 0.07
    intersection_min[1] += 0.07
    intersection_min[2] -= 0.13

    # Shrink the intersection box
    # box_min_shrunk, box_max_shrunk = shrink_box(intersection_min, intersection_max, shrink_factor)
    
    # Generate points in the intersection box
    if seed is not None:
        np.random.seed(seed=seed)
    random_points = np.random.uniform(0, 1, size=(num_points, 3))
    box_dimensions = intersection_max - intersection_min
    scaled_points = random_points * box_dimensions + intersection_min
    
    return scaled_points

def sample_points_on_boarder_in_intersecting_boxes(
    box1_min=np.array([-0.605, -0.533, 0.128]),
    box1_max=np.array([-0.263, -0.363, 0.377]),
    box2_min=np.array([-0.650, -0.569, 0.114]),
    box2_max=np.array([-0.352, 0.369, 0.387]),
    num_points=1,
    seed=42):
    """
    Sample points from intersection of two 3D boxes, keeping x,y fixed at intersection_min
    and only varying z coordinate.
    
    Parameters:
    -----------
    box1_min, box1_max : np.ndarray
        Coordinates of first box
    box2_min, box2_max : np.ndarray
        Coordinates of second box
    num_points : int
        Number of points to sample
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    np.ndarray or None
        Array of sampled points, or None if boxes don't intersect
    """
    # Calculate intersection
    intersection = calculate_box_intersection(box1_min, box1_max, box2_min, box2_max)

    if intersection is None:
        print("The boxes do not intersect!")
        return None
    
    intersection_min, intersection_max = intersection
    
    # Apply fixed margins
    intersection_max[0] -= 0.10  # x max
    intersection_max[1] -= 0.13  # y max
    intersection_max[2] -= 0.20  # z max
    intersection_min[0] -= 0.00  # x min
    intersection_min[1] += 0.33  # y min
    intersection_min[2] += 0.03  # z min
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Create array for sampled points
    sampled_points = np.zeros((num_points, 3))
    
    # Set y coordinates to intersection_min values
    sampled_points[:, 1] = intersection_min[1]  # Fixed y
    
    # Sample random z coordinates
    x_min, x_max = intersection_min[0], intersection_max[0]
    z_min, z_max = intersection_min[2], intersection_max[2]
    sampled_points[:, 0] = np.random.uniform(x_min, x_max, num_points)
    sampled_points[:, 2] = np.random.uniform(z_min, z_max, num_points)
    
    return sampled_points

def plot_3d_points(point1, point2, point3, point4, samples):
    # Create a 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate points
    points = np.array([
        point1,
        point2,
        point3,
        point4
    ])
    
    # Separate x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Plot the points with labels
    scatter = ax.scatter(x, y, z, c='red', marker='o', s=100, label='3D Points')

    # Plot the intersection points
    if samples is not None:
        x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
        scatter = ax.scatter(x, y, z, c='green', marker='x', s=50) #, label='Intersection Points')
    
    # Add labels to each point
    for i, (xi, yi, zi) in enumerate(points):
        ax.text(xi, yi, zi, f'Point {i+1}', color='black')
    
    # Draw two boxes (considering points in pairs)
    # First box: point1 to point2
    # Second box: point3 to point4
    box_pairs = [(0, 1), (2, 3)]
    
    for pair in box_pairs:
        p1, p2 = points[pair[0]], points[pair[1]]
        
        # Create box vertices
        # We'll create a cuboid with the two points at opposite corners
        vertices = np.array([
            p1,  # origin point
            [p2[0], p1[1], p1[2]],  # x changed
            [p2[0], p2[1], p1[2]],  # x and y changed
            [p1[0], p2[1], p1[2]],  # y changed
            [p1[0], p1[1], p2[2]],  # z changed
            [p2[0], p1[1], p2[2]],  # x and z changed
            p2,  # opposite corner point
            [p1[0], p2[1], p2[2]]   # y and z changed
        ])
        
        # Define the edges of the cuboid
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
        ]
        
        # Plot the box edges
        for edge in edges:
            ax.plot3D(
                vertices[edge, 0], 
                vertices[edge, 1], 
                vertices[edge, 2], 
                color='blue', 
                linestyle='--', 
                alpha=0.5
            )
    
    # Customize the plot
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Point Visualization with Boxes')
    
    # Add a grid
    ax.grid(True)
    
    # Add a legend
    ax.legend()
    
    # Adjust the view angle for better visibility
    ax.view_init(elev=20, azim=45)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig("3d_points_with_boxes.png")

def main():
    config = UR5CameraConfigDualRobot()
    T = config.T_O1_O2
    task = config.TASK
    
    # Example boxes
    box1_min = np.concatenate([config.ABS_POSE_LIMIT_LOW_ROBOT_1[task][:3], [1]])
    box1_max = np.concatenate([config.ABS_POSE_LIMIT_HIGH_ROBOT_1[task][:3], [1]])
    box2_min = np.concatenate([config.ABS_POSE_LIMIT_LOW_ROBOT_2[task][:3], [1]])
    box2_max = np.concatenate([config.ABS_POSE_LIMIT_HIGH_ROBOT_2[task][:3], [1]])

    # Print original boxes BEFORE transformation
    print("ORIGINAL BOXES:")
    print("Box 1 Min (before):", box1_min[:3])
    print("Box 1 Max (before):", box1_max[:3])
    print("Box 2 Min (before):", box2_min[:3])
    print("Box 2 Max (before):", box2_max[:3])

    # Evaluate box limits in the correct reference frame
    box2_min = T @ box2_min
    box2_max = T @ box2_max

    # Print boxes AFTER transformation
    print("\nAFTER TRANSFORMATION:")
    print("Box 1 Min:", box1_min[:3])
    print("Box 1 Max:", box1_max[:3])
    print("Box 2 Min:", box2_min[:3])
    print("Box 2 Max:", box2_max[:3])

    # Manually order the boxes 
    box1_min_ordered = np.minimum(box1_min[:3], box1_max[:3])
    box1_max_ordered = np.maximum(box1_min[:3], box1_max[:3])
    box2_min_ordered = np.minimum(box2_min[:3], box2_max[:3])
    box2_max_ordered = np.maximum(box2_min[:3], box2_max[:3])

    print("\nORDERED BOXES:")
    print("Box 1 Min (ordered):", box1_min_ordered)
    print("Box 1 Max (ordered):", box1_max_ordered)
    print("Box 2 Min (ordered):", box2_min_ordered)
    print("Box 2 Max (ordered):", box2_max_ordered)

    # Add detailed intersection check
    def check_intersection(b1_min, b1_max, b2_min, b2_max):
        is_intersecting = not (
            np.any(b1_max < b2_min) or 
            np.any(b2_max < b1_min)
        )
        return is_intersecting

    intersection_result = check_intersection(
        box1_min_ordered, box1_max_ordered, 
        box2_min_ordered, box2_max_ordered
    )
    print("\nDo boxes intersect?", intersection_result)
    
    # Sample points in the intersection
    intersection_points = sample_points_in_intersecting_boxes(
        box1_min[:3], box1_max[:3], box2_min[:3], box2_max[:3], 20, shrink_factor=0.
    )

    intersection_points = sample_points_on_boarder_in_intersecting_boxes(
        box1_min[:3], box1_max[:3], box2_min[:3], box2_max[:3], 20
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

    #plot_3d_points(box1_min[:3], box1_max[:3], box2_min[:3], box2_max[:3], intersection_points)
    plot_3d_points(box1_min_ordered[:3], box1_max_ordered[:3], box2_min_ordered[:3], box2_max_ordered[:3], intersection_points)


if __name__ == "__main__":
    main()