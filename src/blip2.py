import pandas as pd
import numpy as np
import torch

from lavis.models import load_model_and_preprocess


class Blip2:

    def __init__(self,model_name, model_type, device):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        
        self.model, self.vis_process, self.txt_process = load_model_and_preprocess(name=model_name, model_type=model_type, device=device)
    
    def evaluate_caption(self,raw_image):
        image = self.vis_process["eval"](raw_image).unsqueeze(0).to(self.device)
        return model.generate({"image": image})

    def evaluate_QA(self,question, raw_image):
        format_question = f'Question: {question} Answer:'
        image = self.vis_process["eval"](raw_image).unsqueeze(0).to(self.device)
       # print(f"QA = {question}")
        return self.model.generate({"image":image, "prompt": format_question})


