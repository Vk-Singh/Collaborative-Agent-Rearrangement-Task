import pandas as pd
from blip2_agent.dialog_manager import *

class BlipService:
    def __init__(self, port, level, variant):
        self.port = port
        self.level = level
        self.variant = variant
        self.dial_manager = dialog(self.port)


    def get_scene(self, image_path="src/blip2_agent/leader_scene.npy"):
        print("calling scene")
        self.dial_manager.load_image(image_path)

    def get_response(self, follower_message):
        resp =  self.dial_manager.run(follower_message)
        return resp[0]

if __name__ == "__main__":
    srv = BlipService(5000, "l2", "v3")
    srv.get_scene()
    txt= "is there bread on the table?"
    print(srv.get_response(txt))
