import asyncio
import threading
from websockets.asyncio.client import connect
import msgpack
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

async def read_vision_from_server():
    """
    Function used to read the data from the server containing the pose of the boxes in the scene. The unit measure of the output is in meters.

    Keys:
    - space-boxes-box-world2box: pose from the camera frame to the center of the box (exponential coordinates for the orientation)
    """
    messages = []
    async with connect("ws://192.168.1.240:7777") as websocket:
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
        self.size = []
        self.start_idx = 0
        
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Start the async loop in a separate thread
        self.thread = threading.Thread(target=self._run_async_loop)
        self.thread.daemon = True
        self.thread.start()

        self.hz = 30

        self.last_heartbeat = None
        self.HEARTBEAT_INTERVAL = 30  # seconds
        self.MAX_RECONNECT_ATTEMPTS = 10000

    def compute_3d_ema(self, data, alpha=0.1):
        """
        Compute the Exponential Moving Average (EMA) of the data
        
        Parameters:
        -----------
        data : np.ndarray
            Array of data points to compute the EMA
        alpha : float
            Smoothing factor for the EMA. Higher alpha gives more weight to recent data.
        
        Returns:
        --------
        ema: np.ndarray
            Array of EMA values
        """
        if not data.any():
            return np.zeros(3)
        
        assert 0 < alpha < 1, "Smoothing factor must be between 0 and 1"
        assert data.shape[1] == 3, "Data must be a 3D vector!"

        ema = np.zeros_like(data)
        ema[self.start_idx] = data[self.start_idx]
        for i in range(self.start_idx + 1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        self.start_idx = len(data) - 1
        return ema[-1]
    
    def compute_6d_ema(self, data, alpha=0.1):
        """
        Compute the Exponential Moving Average (EMA) of the data
        
        Parameters:
        -----------
        data : np.ndarray
            Array of data points to compute the EMA
        alpha : float
            Smoothing factor for the EMA
        
        Returns:
        --------
        ema: np.ndarray
            Array of EMA values
        """
        if not data.any():
            return np.zeros(6)
        
        assert 0 < alpha < 1, "Smoothing factor must be between 0 and 1"
        assert data.shape[1] == 6, "Data must be a 6D vector"
        
        ema = np.zeros_like(data)
        ema[self.start_idx] = data[self.start_idx]
        for i in range(self.start_idx + 1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

        self.start_idx = len(data) - 1
        return ema[-1]
    
    def _run_async_loop(self):
        """Run the async event loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._read_vision_from_server())
        except Exception as e:
            print(f"Error in vision server loop: {e}")

    async def receive_message(self, websocket):
        try:
            return await asyncio.wait_for(websocket.recv(), timeout=1/self.hz)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {1/self.hz} seconds")
        except Exception as e:
            print(f"WebSocket receive error: {e}")
            raise
        
    async def _send_heartbeat(self, websocket):
        """Send heartbeat to keep connection alive"""
        try:
            await websocket.send(msgpack.packb("heartbeat"))
            self.last_heartbeat = datetime.now()
        except Exception as e:
            print(f"Heartbeat error: {e}")
            raise
    
    async def _read_vision_from_server(self):
        """
        Async function to read data from the server containing box pose
        """
        reconnect_attempts = 0
        
        while not self.stop_event.is_set():
            try:
                async with connect(self.ip_address) as websocket:
                    reconnect_attempts = 0  # Reset counter on successful connection
                    print("Connected to server")
                    while not self.stop_event.is_set():
                        if (not self.last_heartbeat or 
                            (datetime.now() - self.last_heartbeat).seconds >= self.HEARTBEAT_INTERVAL):
                            await self._send_heartbeat(websocket)
                                
                            if self.hz > 0:
                                await asyncio.sleep(1/self.hz)
                        try:
                            raw_message = await self.receive_message(websocket)
                            message = msgpack.unpackb(raw_message)
                            # await websocket.send("a")
                        
                            # print("Message received")
                            # Safely update the message with a lock
                            with self.state_lock:
                                self.pos.append(message['space'][0]['boxes'][
                                    list(message['space'][0]['boxes'].keys())[0]
                                ]['world2box']['pos'])
                                self.orient.append(message['space'][0]['boxes'][
                                    list(message['space'][0]['boxes'].keys())[0]
                                ]['world2box']['rot'])
                                self.size = message['space'][0]['boxes'][
                                    list(message['space'][0]['boxes'].keys())[0]
                                ]['size']
                        except TimeoutError:
                            # print("Timeout error, continuing...")
                            self.last_heartbeat = None
                            continue

            except Exception as e:
                # if the box is not detected, the last self.pos is kept (dict key doesn't exist)
                # print(f"Connection error: {e}")
                reconnect_attempts += 1
                
                if reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
                    print("Max reconnection attempts reached")
                    break
                    
                await asyncio.sleep(1)  # Wait before reconnecting
    
    def get_box_position(self):
        """
        Thread-safe method to get the current box position
        """
        with self.state_lock:
            return np.array(self.pos)
        
    def get_box_orientation(self):
        """
        Thread-safe method to get the current box orientation expressed in angle-axis representation
        """
        with self.state_lock:
            return np.array(self.orient)
        
    def get_box_pose(self):
        """
        Thread-safe method to get the current box pose
        """
        with self.state_lock:
            return np.concatenate([np.array(self.pos), np.array(self.orient)], axis=1)
        
    def get_box_size(self):
        """
        Thread-safe method to get the current box size
        """
        with self.state_lock:
            return np.array(self.size)
        
    def clear_data(self):
        """
        Thread-safe method to clear the data
        """
        with self.state_lock:
            self.pos = []
            self.orient = []
            self.size = []
            self.start_idx = 0
    
    def stop(self):
        """
        Method to gracefully stop the async loop
        """
        self.stop_event.set()
        self.thread.join()

    def __del__(self):
        """
        Destructor to ensure thread cleanup
        """
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.stop()

if __name__ == "__main__":
    # messages = asyncio.run(read_vision_from_server())
    # print(messages)
    box_pose_estimation = BoxPoseEstimation("ws://192.168.1.240:7777")

    import time
    start = time.time()
    while box_pose_estimation.get_box_position() == []:
        continue
    end = time.time()
    print(f"Time to get first message: {end - start}")
    while True:
        
        try:
            pos = np.array(box_pose_estimation.get_box_position())
            orient = np.array(box_pose_estimation.get_box_orientation())
            # print(f"Position: {pos}")
            # print(f"Orientation: {orient}")
        except KeyboardInterrupt:
            box_pose_estimation.stop()
            break