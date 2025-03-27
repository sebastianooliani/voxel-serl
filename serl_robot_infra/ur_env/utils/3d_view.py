import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.transform as transform
from matplotlib.widgets import Button

def plot_marker_boards(json_data):
    """
    Plot marker boards from a JSON object with interactive 3D view.
    
    Parameters:
    json_data (dict): Dictionary containing marker board data
    """
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        marker_data = json.loads(json_data)
    else:
        marker_data = json_data
    
    # Create a 3D plot with interactive viewing
    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')
    
    # Enable interactive mode
    plt.ion()
    
    # Define colors for different boards
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # For each object in the data
    for object_id, object_data in marker_data.items():
        if 'board2ds' in object_data:
            print(f"Processing {object_id}...")
            
            for i, board in enumerate(object_data['board2ds']):
                # Get board properties
                marker_length = board['markerLength']
                marker_spacing = board['markerSpacing']
                num_columns = board['numColumns']
                num_rows = board['numRows']
                tag_id_start = board['tagIdStart']
                
                # Get position and rotation
                pos = board['three2two']['pos']
                rot_vec = board['three2two']['rot']  # Rotation vector
                
                # Board color
                color = colors[i % len(colors)]
                
                # Calculate board dimensions
                board_width = num_columns * marker_length + (num_columns - 1) * marker_spacing
                board_height = num_rows * marker_length + (num_rows - 1) * marker_spacing
                
                # Plot the board as a rectangle
                plot_board(ax, pos, rot_vec, board_width, board_height, color, f"{object_id}-Board {i+1}")
                
                # Plot the individual markers on the board
                plot_markers(ax, pos, rot_vec, marker_length, marker_spacing, num_rows, num_columns, tag_id_start, color)
    
    # Origin point
    ax.scatter(0, 0, 0, s=200, c='black', marker='*', label='Origin')
    
    # Add coordinate axes for better orientation
    add_coordinate_axes(ax)
    
    # Set labels and title
    ax.set_xlabel('X axis (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y axis (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z axis (m)', fontsize=12, labelpad=10)
    ax.set_title(f'Marker Boards for {list(marker_data.keys())[0]}', fontsize=14, pad=20)
    
    # Create a legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Adjust axis limits for better visualization
    max_range = 0.3  # Adjust based on your data
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Add viewing angles buttons
    add_view_buttons(plt.gcf(), ax)
    
    # Enable grid
    ax.grid(True)
    
    # Add a text box with usage instructions
    instruction_text = "Instructions:\n• Left click and drag to rotate\n• Right click and drag to pan\n• Scroll to zoom\n• Use view buttons for preset angles"
    plt.figtext(0.02, 0.02, instruction_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Show the plot (will be interactive)
    plt.tight_layout()
    plt.show()
    
    return ax

def add_coordinate_axes(ax, length=0.1):
    """Add coordinate axes arrows to help with orientation."""
    # Origin point (already plotted as a star, this is just for the arrows)
    origin = [0, 0, 0]
    
    # X axis (red)
    ax.quiver(origin[0], origin[1], origin[2], length, 0, 0, color='red', linewidth=2, arrow_length_ratio=0.1)
    ax.text(length*1.1, 0, 0, "X", color='red', fontsize=12)
    
    # Y axis (green)
    ax.quiver(origin[0], origin[1], origin[2], 0, length, 0, color='green', linewidth=2, arrow_length_ratio=0.1)
    ax.text(0, length*1.1, 0, "Y", color='green', fontsize=12)
    
    # Z axis (blue)
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, length, color='blue', linewidth=2, arrow_length_ratio=0.1)
    ax.text(0, 0, length*1.1, "Z", color='blue', fontsize=12)

def add_view_buttons(fig, ax):
    """Add buttons for different viewing angles."""
    # Button positions
    button_width = 0.08
    button_height = 0.05
    button_padding = 0.01
    
    # Create button for top view
    ax_top = fig.add_axes([0.2, 0.01, button_width, button_height])
    btn_top = Button(ax_top, 'Top View')
    btn_top.on_clicked(lambda event: set_view(ax, 'top'))
    
    # Create button for front view
    ax_front = fig.add_axes([0.2 + button_width + button_padding, 0.01, button_width, button_height])
    btn_front = Button(ax_front, 'Front View')
    btn_front.on_clicked(lambda event: set_view(ax, 'front'))
    
    # Create button for side view
    ax_side = fig.add_axes([0.2 + 2 * (button_width + button_padding), 0.01, button_width, button_height])
    btn_side = Button(ax_side, 'Side View')
    btn_side.on_clicked(lambda event: set_view(ax, 'side'))
    
    # Create button for isometric view
    ax_iso = fig.add_axes([0.2 + 3 * (button_width + button_padding), 0.01, button_width, button_height])
    btn_iso = Button(ax_iso, 'Isometric')
    btn_iso.on_clicked(lambda event: set_view(ax, 'isometric'))

def set_view(ax, view_type):
    """Set the view based on the button clicked."""
    if view_type == 'top':
        ax.view_init(elev=90, azim=-90)
    elif view_type == 'front':
        ax.view_init(elev=0, azim=-90)
    elif view_type == 'side':
        ax.view_init(elev=0, azim=0)
    elif view_type == 'isometric':
        ax.view_init(elev=30, azim=45)
    plt.draw()

def apply_rotation_vector(point, rot_vec):
    """
    Apply rotation vector to a point.
    
    Parameters:
    point (list): The point [x, y, z] to rotate
    rot_vec (list): The rotation vector [rx, ry, rz]
    
    Returns:
    list: The rotated point [x', y', z']
    """
    # Convert point to numpy array
    point_np = np.array(point)
    
    # Calculate the angle (magnitude of the rotation vector)
    angle = np.linalg.norm(rot_vec)
    
    # If angle is close to zero, no rotation needed
    if angle < 1e-10:
        return point
    
    # Calculate the axis (normalized rotation vector)
    axis = np.array(rot_vec) / angle
    
    # Create a rotation object
    rotation = transform.Rotation.from_rotvec(axis * angle)
    
    # Apply the rotation to the point
    rotated_point = rotation.apply(point_np)
    
    return rotated_point.tolist()

def plot_board(ax, pos, rot_vec, width, height, color, label):
    """Plot a rectangular board with given position and rotation."""
    # Define the corners of the board (centered at origin, before rotation/translation)
    half_width = width / 2
    half_height = height / 2
    
    # Define corners of the board
    corners = [
        [-half_width, -half_height, 0],
        [half_width, -half_height, 0],
        [half_width, half_height, 0],
        [-half_width, half_height, 0],
        [-half_width, -half_height, 0]  # Close the rectangle
    ]
    
    # Apply rotation and translation to each corner
    transformed_corners = []
    for corner in corners:
        rotated = apply_rotation_vector(corner, rot_vec)
        translated = [rotated[0] + pos[0], rotated[1] + pos[1], rotated[2] + pos[2]]
        transformed_corners.append(translated)
    
    # Extract x, y, z coordinates for plotting
    x_vals = [corner[0] for corner in transformed_corners]
    y_vals = [corner[1] for corner in transformed_corners]
    z_vals = [corner[2] for corner in transformed_corners]
    
    # Plot the board outline
    ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=2, alpha=0.7, label=label)
    
    # Plot the center of the board
    ax.scatter(pos[0], pos[1], pos[2], color=color, s=50)
    
    # Fill the board with a transparent surface for better visualization
    vertices = np.array([x_vals[:-1], y_vals[:-1], z_vals[:-1]]).T
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=color, alpha=0.2)

def plot_markers(ax, board_pos, board_rot, marker_length, marker_spacing, num_rows, num_columns, tag_id_start, color):
    """Plot individual markers on the board."""
    half_board_width = (num_columns * marker_length + (num_columns - 1) * marker_spacing) / 2
    half_board_height = (num_rows * marker_length + (num_rows - 1) * marker_spacing) / 2
    
    tag_id = tag_id_start
    
    for row in range(num_rows):
        for col in range(num_columns):
            # Calculate marker position relative to board center
            marker_x = -half_board_width + col * (marker_length + marker_spacing) + marker_length / 2
            marker_y = -half_board_height + row * (marker_length + marker_spacing) + marker_length / 2
            marker_z = 0
            
            # Apply board rotation and translation
            marker_pos = apply_rotation_vector([marker_x, marker_y, marker_z], board_rot)
            marker_pos = [
                marker_pos[0] + board_pos[0],
                marker_pos[1] + board_pos[1],
                marker_pos[2] + board_pos[2]
            ]
            
            # Plot marker
            ax.scatter(marker_pos[0], marker_pos[1], marker_pos[2], color=color, s=30, alpha=0.8)
            
            # Add marker ID label
            ax.text(marker_pos[0], marker_pos[1], marker_pos[2], f"{tag_id}", fontsize=8, ha='center', va='center')
            
            # Draw marker outline as a small square
            plot_marker_outline(ax, marker_pos, board_rot, marker_length, color)
            
            tag_id += 1

def plot_marker_outline(ax, center_pos, board_rot, size, color):
    """Draw a square outline for each marker."""
    half_size = size / 2
    
    # Define corners of the marker (centered at origin, before rotation/translation)
    corners = [
        [-half_size, -half_size, 0],
        [half_size, -half_size, 0],
        [half_size, half_size, 0],
        [-half_size, half_size, 0],
        [-half_size, -half_size, 0]  # Close the rectangle
    ]
    
    # First apply the same rotation as the board
    rotated_corners = [apply_rotation_vector(corner, board_rot) for corner in corners]
    
    # Then translate to the marker position
    transformed_corners = [
        [corner[0] + center_pos[0], corner[1] + center_pos[1], corner[2] + center_pos[2]]
        for corner in rotated_corners
    ]
    
    # Extract x, y, z coordinates for plotting
    x_vals = [corner[0] for corner in transformed_corners]
    y_vals = [corner[1] for corner in transformed_corners]
    z_vals = [corner[2] for corner in transformed_corners]
    
    # Plot the marker outline
    ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=1, alpha=0.5)


# Example usage
if __name__ == "__main__":
    # Your JSON data
    json_data = {
        "box_54":{
            "board2ds": [
                {
                    "markerLength": 0.038,
                    "markerSpacing": 0.009,
                    "numColumns": 2,
                    "numRows": 1,
                    "tagIdStart": 148,
                    "three2two": {
                        "pos": [
                            -0.106,
                            -0.1505,
                            -0.0525
                        ],
                        "rot": [
                            0.0,
                            0.0,
                            0.0
                        ]
                    }
                },
                {
                    "markerLength": 0.038,
                    "markerSpacing": 0.009,
                    "numColumns": 2,
                    "numRows": 1,
                    "tagIdStart": 150,
                    "three2two": {
                        "pos": [
                            0.105,
                            -0.150,
                            -0.0465
                        ],
                        "rot": [
                            0.0,
                            -1.57,
                            0.0
                        ]
                    }
                },
                {
                    "markerLength": 0.038,
                    "markerSpacing": 0.009,
                    "numColumns": 2,
                    "numRows": 1,
                    "tagIdStart": 152,
                    "three2two": {
                        "pos": [
                            0.106,
                            -0.1505,
                            0.0525
                        ],
                        "rot": [
                            0.0,
                            3.141593,
                            0.0
                        ]
                    }
                },
                {
                    "markerLength": 0.038,
                    "markerSpacing": 0.009,
                    "numColumns": 2,
                    "numRows": 1,
                    "tagIdStart": 154,
                    "three2two": {
                        "pos": [
                            -0.105,
                            -0.150,
                             0.0465
                        ],
                        "rot": [
                            0.0,
                            1.57,
                            0.0
                        ]
                    }
                },
                {
                    "markerLength": 0.038,
                    "markerSpacing": 0.009,
                    "numColumns": 2,
                    "numRows": 1,
                    "tagIdStart": 156,
                    "three2two": {
                        "pos": [
                            0.1055,
                            -0.150,
                            -0.0525
                        ],
                        "rot": [
                            0.0,
                            2.22,
                            2.22
                        ]
                    }
                },
                {
                    "markerLength": 0.038,
                    "markerSpacing": 0.009,
                    "numColumns": 2,
                    "numRows": 1,
                    "tagIdStart": 158,
                    "three2two": {
                        "pos": [
                            0.1055,
                            0.150,
                            0.0525
                        ],
                        "rot": [
                            0.0,
                            -2.22,
                            2.22
                        ]
                    }
                }
            ]
        }
    }
    # Plot and return the axis for further manipulation if needed
    ax = plot_marker_boards(json_data)
    
    # Keep the plot window open until the user closes it
    plt.show(block=True)