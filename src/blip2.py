
import numpy as np
import torch

from lavis.models import load_model_and_preprocess


class Blip2:

    def __init__(self,model_name, model_type, device):
        """
        Initialize a Blip2 model.

        Parameters
        ----------
        model_name : str
            The name of the model.
        model_type : str
            The type of the model.
        device : str
            The device to run the model on.

        Attributes
        ----------
        model_name : str
            The name of the model.
        model_type : str
            The type of the model.
        device : str
            The device to run the model on.
        max_length : int
            The maximum length of the generated text.
        length_penalty : float
            The length penalty for the generated text.
        repetition_penalty : float
            The repetition penalty for the generated text.
        temperature : float
            The temperature for the generated text.
        model : torch.nn.Module
            The PyTorch model.
        vis_process : dict
            The visualization process.
        txt_process : dict
            The text process.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.max_length = 30
        self.length_penalty = 1
        self.repetition_penalty = 1.5
        self.temperature=1
        
        self.model, self.vis_process, self.txt_process = load_model_and_preprocess(name=model_name, model_type=model_type, device=device)
    
    def evaluate_caption(self,raw_image):
        """
        Evaluate a caption for a given image.

        Parameters
        ----------
        raw_image : PIL.Image
            The image to generate a caption for.

        Returns
        -------
        generated_text : str
            The generated caption.
        """
        image = self.vis_process["eval"](raw_image).unsqueeze(0).to(self.device)
        return self.model.generate({"image": image})

    def evaluate_QA(self,question, raw_image):
        """
        Evaluate a question for a given image.

        Parameters
        ----------
        question : str
            The question to evaluate.
        raw_image : PIL.Image
            The image to evaluate the question for.

        Returns
        -------
        generated_text : str
            The generated answer.
        """
        format_question = f'Question: {question} Answer:'
        image = self.vis_process["eval"](raw_image).unsqueeze(0).to(self.device)
        #return self.model.generate({"image":image, "prompt": format_question}, "do_sample":False,"num_beams":5, "max_length":256, "min_length":1,
        #                                        "top_p":0.9,"repetition_penalty":1.5,"length_penalty":1.0,"temperature":1)
        #return self.model.generate({"image": image, "prompt": format_question}, use_nucleus_sampling=True, 
        #             max_length=self.max_length, length_penalty=self.length_penalty, repetition_penalty=self.repetition_penalty, temperature=self.temperature)
        return self.model.generate({"image":image, "prompt": format_question}, use_nucleus_sampling=False, num_beams=5, max_length=256, min_length=1,
                                                top_p=0.9,repetition_penalty=1.5,length_penalty=1.0,temperature=1)
        
