import pandas as pd
from blip2_agent.dialog_manager import *

class BlipService:
    def __init__(self, port, level, variant):
        """
        Initialize the BlipService object.

        Args:
            port (int): The TCP port used for communication with the client.
            level (str): The level of the game (e.g. "easy", "medium", "hard").
            variant (str): The variant of the game (e.g. "standard", "object_recognition").

        Returns:
            None
        """
        self.port = port
        self.level = level
        self.variant = variant
        self.dial_manager = dialog(self.port)


    def get_scene(self, image_path="src/blip2_agent/leader_scene.npy"):
        """
        Load the scene image and initialize the dialog manager.

        Args:
            image_path (str): The path to the scene image (default: "src/blip2_agent/leader_scene.npy").

        Returns:
            None
        """
        print("calling scene")
        self.dial_manager.load_image(image_path)

    def get_response(self, follower_message):
        """
        Get a response from the dialog manager given the follower's message.

        Args:
            follower_message (str): The message from the follower.

        Returns:
            str: The response from the leader.
        """
        resp =  self.dial_manager.run(follower_message)
        return resp[0]

if __name__ == "__main__":
    srv = BlipService(5000, "l2", "v3")
    srv.get_scene()
    txt= "is there bread on the table?"
    print(srv.get_response(txt))
