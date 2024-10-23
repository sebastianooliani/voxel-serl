import threading
import pyspacemouse
import numpy as np
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provide
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        # open() returns a device object if the device was opened successfully
        self.mouse = pyspacemouse.open()

        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 0]}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        while True:
            state = self.mouse.read()
            with self.state_lock:
                self.latest_data["action"] = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )  # spacemouse axis matched with robot base frame
                self.latest_data["buttons"] = state.buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]
        

class TwoSpaceMiceExperts():
    """
    This class provides an interface to connect and read the inputs from two SpaceMice.
    If you have multiple 3Dconnexion devices, you can use the object-oriented API to access them individually. 

    Each object has the following API, which functions exactly as the above API, but on a per-device basis:
        dev.open()          Opens the connection (this is always called by the module-level open command, 
                            so you should not need to use it unless you have called close())
        dev.read()          Return the state of the device as namedtuple [t,x,y,z,roll,pitch,yaw,button]
        dev.close()         Close this device
        dev.set_led(state)  Set the state of the LED on the device to on (True) or off (False)
    """
    def __init__(self):
        self.left_arm = pyspacemouse.open(DeviceNumber=1)
        # print(self.left_arm.device, self.left_arm.name)
        self.right_arm = pyspacemouse.open(DeviceNumber=2)
        # print(self.right_arm.device, self.right_arm.name)
        
        self.state_lock = threading.Lock()
        self.latest_data = {"action_1": np.zeros(6), "buttons_1": [0, 0],
                            "action_2": np.zeros(6), "buttons_2": [0, 0]}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()
        

    def _read_spacemouse(self):
        while True:
            # IP: "...66"
            state_1 = self.left_arm.read()
            # IP: "...33"
            # breakpoint()
            state_2 = self.right_arm.read()

            with self.state_lock:
                self.latest_data["action_1"] = np.array(
                    [-state_1.y, state_1.x, state_1.z, -state_1.roll, -state_1.pitch, -state_1.yaw]
                )
                self.latest_data["buttons_1"] = state_1.buttons
                self.latest_data["action_2"] = np.array(
                    [-state_2.y, state_2.x, state_2.z, -state_2.roll, -state_2.pitch, -state_2.yaw]
                )
                self.latest_data["buttons_2"] = state_2.buttons

    def get_action(self) -> Tuple[np.ndarray, list, np.ndarray, list]:
        with self.state_lock:
            return self.latest_data["action_1"], self.latest_data["buttons_1"], self.latest_data["action_2"], self.latest_data["buttons_2"]