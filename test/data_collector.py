import pandas as pd
import os
import sys
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter.simpledialog import askstring
from ithor.ithor_controller import IthorController


class DataCollection:
    
    def __init__(self, input_file):
        self.leader_scenes = None
        self.input_file = input_file
        self.scenes = None    
        self.root = tk.Tk()


    def load_all_scenes(self):
        with open(self.input_file, encoding="utf-8") as json_input:
            self.scenes = json.load(json_input)


    def setup_scene(self, level, variant):
        self.ithor_controller = IthorController()
        self.leader_scene = self.scenes[level][variant]["leader"]
        self.ithor_controller.init_scene(pos=[0.25,1,0], rot=270, horizon=70)
        self.ithor_controller.place_assets(self.leader_scene)


    def save_image(self, file_path):
        img_arr = self.ithor_controller.snapshot_frame()
       # img = cv2.cvtColor(np.asarray(img_arr), cv2.COLOR_BGR2RGB)
        #img = cv2.imread(np.asarray(img_arr))
        print(img_arr.shape)
        crop_img = img_arr[400:1200, 50:1500].copy()
        # Check the naming convention of the scenes
        cv2.imwrite(file_path, crop_img)
       # cv2.imshow("Image", crop_img)
       # cv2.waitKey(0)


    def setup_caption(self):
        prompt = askstring("Input", "Input a caption for the image...", parent=self.root)
        print(prompt)



    def run(self,max_lvl=0, max_variant=0, image_save_path="src/blip2_agent/data/test/",caption_save_path=None ):

        self.load_all_scenes()

        for l in range(max_lvl + 1):
            for v in range(1, max_variant + 1):
                data_collector.setup_scene(f"l{l}",f"v{v}")
                data_collector.save_image(f"{image_save_path}{l}_{v}.jpg")
                self.setup_caption()
                self.ithor_controller.stop()


def pause():
    temp_input = input("Enter to close application......")


if __name__ == "__main__":
    data_collector = DataCollection(f"src/ithor/scene_configs2023-06-24.json")
    l = 5
    v = 50
    data_collector.run(l,v)
