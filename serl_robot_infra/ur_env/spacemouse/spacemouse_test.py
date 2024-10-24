""" Test the spacemouse output. """
import time
import numpy as np
from ur_env.spacemouse.spacemouse_expert import SpaceMouseExpert, TwoSpaceMiceExperts
from ur_env.envs.wrappers import TwoSpacemiceIntervention
import pyspacemouse



def test_spacemouse():
    """Test the SpaceMouseExpert class.

    This interactive test prints the action and buttons of the spacemouse at a rate of 10Hz.
    The user is expected to move the spacemouse and press its buttons while the test is running.
    It keeps running until the user stops it.

    """
    spacemouse = SpaceMouseExpert()
    with np.printoptions(precision=3, suppress=True):
        while True:
            action, buttons = spacemouse.get_action()
            print(f"Spacemouse action: {action}, buttons: {buttons}")
            time.sleep(0.1)

def test_two_spacemice():
    """Test the TwoSpaceMiceExperts class.

    This interactive test prints the actions and buttons of the two spacemice at a rate of 10Hz.
    The user is expected to move the spacemice and press their buttons while the test is running.
    It keeps running until the user stops it.

    """
    spacemouse_1 = SpaceMouseExpert(DeviceNumber=0)
    spacemouse_2 = SpaceMouseExpert(DeviceNumber=3)
    with np.printoptions(precision=3, suppress=True):
        while True:
            action_1, buttons_1 = spacemouse_1.get_action()
            action_2, buttons_2 = spacemouse_2.get_action()
            
            print(f"Left arm action: {action_1}, buttons: {buttons_1}")
            print(f"Right arm action: {action_2}, buttons: {buttons_2}")
            time.sleep(0.1)

def main():
    """Call spacemouse test."""
    test_spacemouse()
    # test_two_spacemice()

if __name__ == "__main__":
    main()
