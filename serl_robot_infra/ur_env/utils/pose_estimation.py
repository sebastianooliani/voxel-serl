import asyncio
from websockets.asyncio.client import connect
import msgpack
import matplotlib.pyplot as plt
import numpy as np

async def read_vision_from_server():
    """
    Function used to read the data from the server containing the pose of the boxes in the scene. The unit measure of the output is in meters.

    Keys:
    - space-boxes-box-world2box: pose from the camera frame to the center of the box (exponential coordinates for the orientation)
    """
    messages = []
    async with connect("ws://192.168.1.204:7777") as websocket:
        while True:
            message = msgpack.unpackb(await websocket.recv())
            print(message)
            # send message to mantain the connection alive
            await websocket.send("a")

            if len(messages) < 500:
                messages.append(np.array(message['space'][0]['boxes'][list(message['space'][0]['boxes'].keys())[0]]['world2box']['pos']))
            else:
                break

    return messages

def plot_vector_axes(vectors, output_path=None):
    """
    Create three separate plots for x, y, and z components of 3D vectors.
    
    Parameters:
    -----------
    vectors : list or numpy.ndarray
        List of 3D vectors, where each vector is [x, y, z]
    """
    # Convert to numpy array for easier indexing
    vectors = np.array(vectors)
        
    # Create a figure with three subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 24))
    
    # X-axis values plot
    ax1.plot(vectors[:, 0], label='X Component')
    ax1.set_title('X Component')
    ax1.set_xlabel('Vector Index')
    ax1.set_ylabel('X Value')
    ax1.grid(True)
    ax1.legend()
    
    # Y-axis values plot
    ax2.plot(vectors[:, 1], label='Y Component', color='green')
    ax2.set_title('Y Component')
    ax2.set_xlabel('Vector Index')
    ax2.set_ylabel('Y Value')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout and display
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        plt.close(fig)
    else:
        # Use non-interactive backend
        
        plt.savefig('vector_components.png')
        plt.close(fig)

if __name__ == "__main__":
    messages = asyncio.run(read_vision_from_server())

    plot_vector_axes(messages)