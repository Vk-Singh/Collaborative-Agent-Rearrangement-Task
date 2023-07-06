import pandas as pd
import numpy as np
from PIL import Image
from blip2_agent.classify_model import *
from blip2_agent.blip2 import *

class dialog:

    def __init__(self,user_id, model_name="blip2_t5", model_type="pretrain_flant5xl", model_location=None):
        self.user_id = user_id
        self.model_type = model_type
        self.model_name = model_name
        self.model_location = model_location
        self.image = None
        self.blip_model = None
        self.classifier = None
        self.current_obj = None
        self.current_obj_status = False
        self.label2id = {"NEXT_STEP": 0, "SUCCESS":1, "BLIP":2}
        self.dialog_dict= {}

        self.load_blip_model()

    def load_blip_model(self):
        self.blip_model = Blip2(self.model_name, self.model_type,"cpu")


    def load_image(self, leader_image):
        img = np.load(leader_image)
        self.image = Image.fromarray(img)
        self.visualize_scene()


    def visualize_scene(self):
        ques = "Question: List the objects on the table top"
        print(ques)
        resp = self.blip_model.evaluate_QA(ques, self.image)
        resp_text = resp[0].replace('a ', '')
        resp_text = resp_text.replace('and ', '')
        obj_list = resp_text.split(',')
        print(obj_list)

        self.dialog_dict = {key: False for key in obj_list}
        print(self.dialog_dict)

    def get_next_obj(self):
        for obj, status in self.dialog_dict.items():
                if status == False and obj != self.current_obj:
                    self.current_obj = obj
                    self.current_obj_status = False
                    break

        return self.current_obj

    
    def  blip_generate_instruction(self, obj):
        text = f"say something about {obj}"
        response = self.blip_model.evaluate_QA(text, self.image)
        return response

    
    def blip_generate_QA(self, text):
        response = self.blip_model.evaluate_QA(text, self.image)
        return response


    def classify_text(self, text):
        self.classifier = TextClassifier(self.label2id)
        resp = self.classifier.inference(text)
        return resp[0]["label"]


    def run(self, text):

        cls = self.classify_text(text)
        resp =None
        print (f"cls ==== {cls}")
        if cls =="LABEL_1":
            print("Success. Get new instruction")
            self.dialog_dict[self.current_obj] = True
            obj = self.get_next_obj()
            resp = self.blip_generate_instruction(obj)

        elif cls =="LABEL_0":
            obj = self.get_next_obj()
            resp = self.blip_generate_instruction(obj)
        
        elif cls =="LABEL_2":
            resp = self.blip_generate_QA(text)
        
        return resp
