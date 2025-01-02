import asyncio
import threading
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
    async with connect("ws://192.168.1.184:7777") as websocket:
        while True:
            message = msgpack.unpackb(await websocket.recv())
            # box_position = message['space'][0]['boxes'][list(message['space'][0]['boxes'].keys())[0]]['world2box']['pos']
            # print(f"original frame {box_position}")
            # box_position = np.array([[-1,  0,  0],
            #                         [ 0,  0, -1],
            #                         [ 0, -1,  0]]) @ np.array(box_position)
            # print(f"rotated frame {box_position}")
            # send message to mantain the connection alive
            print(message)
            await websocket.send("a")

            # if len(messages) < 500:
            #     messages.append(np.array(message['space'][0]['boxes'][list(message['space'][0]['boxes'].keys())[0]]['world2box']['pos']))
            # else:
            #     break

    # return messages

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

class BoxPoseEstimation:
    def __init__(self, ip_address: str):
        self.ip_address = ip_address
        self.pos = []
        self.orient = []
        
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Start the async loop in a separate thread
        self.thread = threading.Thread(target=self._run_async_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def _run_async_loop(self):
        """Run the async event loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._read_vision_from_server())
        except Exception as e:
            print(f"Error in vision server loop: {e}")
    
    async def _read_vision_from_server(self):
        """
        Async function to read data from the server containing box pose
        """
        while not self.stop_event.is_set():
            try:
                async with connect(self.ip_address) as websocket:
                    message = msgpack.unpackb(await websocket.recv())
                    
                    # Safely update the message with a lock
                    with self.state_lock:
                        self.pos = message['space'][0]['boxes'][
                            list(message['space'][0]['boxes'].keys())[0]
                        ]['world2box']['pos']
                        self.orient = message['space'][0]['boxes'][
                            list(message['space'][0]['boxes'].keys())[0]
                        ]['world2box']['orient']
                    
                    await websocket.send("a")
                    
            except Exception as e:
                print(f"Error reading from vision server: {e}")
                await asyncio.sleep(1)  # Prevent tight error loop
    
    def get_box_position(self):
        """
        Thread-safe method to get the current box position
        """
        with self.state_lock:
            return np.array(self.pos)
        
    def get_box_orientation(self):
        """
        Thread-safe method to get the current box orientation
        """
        with self.state_lock:
            return np.array(self.orient)
    
    def stop(self):
        """
        Method to gracefully stop the async loop
        """
        self.stop_event.set()
        self.thread.join()

if __name__ == "__main__":
    messages = asyncio.run(read_vision_from_server())