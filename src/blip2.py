
import numpy as np
import torch

from lavis.models import load_model_and_preprocess


class Blip2:

    def __init__(self,model_name, model_type, device):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.max_length = 30
        self.length_penalty = 1
        self.repetition_penalty = 1.5
        self.temperature=1
        
        self.model, self.vis_process, self.txt_process = load_model_and_preprocess(name=model_name, model_type=model_type, device=device)
    
    def evaluate_caption(self,raw_image):
        image = self.vis_process["eval"](raw_image).unsqueeze(0).to(self.device)
        return self.model.generate({"image": image})

    def evaluate_QA(self,question, raw_image):
        format_question = f'Question: {question} Answer:'
        image = self.vis_process["eval"](raw_image).unsqueeze(0).to(self.device)
        #return self.model.generate({"image":image, "prompt": format_question}, "do_sample":False,"num_beams":5, "max_length":256, "min_length":1,
        #                                        "top_p":0.9,"repetition_penalty":1.5,"length_penalty":1.0,"temperature":1)
        #return self.model.generate({"image": image, "prompt": format_question}, use_nucleus_sampling=True, 
        #             max_length=self.max_length, length_penalty=self.length_penalty, repetition_penalty=self.repetition_penalty, temperature=self.temperature)
        return self.model.generate({"image":image, "prompt": format_question}, use_nucleus_sampling=False, num_beams=5, max_length=256, min_length=1,
                                                top_p=0.9,repetition_penalty=1.5,length_penalty=1.0,temperature=1)
        
