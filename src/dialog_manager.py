import pandas as pd
import numpy as np
from PIL import Image
from blip2_agent.classify_model import *
from blip2_agent.blip2 import *

class dialog:

    def __init__(self,user_id, model_name="blip2_t5", model_type="pretrain_flant5xl", model_location=None):
        """
        Initialize the dialog manager.

        Parameters
        ----------
        user_id : str
            The id of the user.
        model_name : str, optional
            The name of the model. Defaults to "blip2_t5".
        model_type : str, optional
            The type of the model. Defaults to "pretrain_flant5xl".
        model_location : str, optional
            The location of the model. Defaults to None.

        Attributes
        ----------
        user_id : str
            The id of the user.
        model_type : str
            The type of the model.
        model_name : str
            The name of the model.
        model_location : str
            The location of the model.
        image : PIL.Image
            The current image.
        blip_model : blip2.Blip2
            The current blip model.
        classifier : torch.nn.Module
            The current classifier.
        current_obj : str
            The current object the user is referring to.
        current_obj_status : bool
            The status of the current object.
        label2id : dict
            The mapping of labels to ids.
        dialog_dict : dict
            The dictionary of the dialog.
        """
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
        """
        Load the blip model.

        This method loads the blip model and set it to self.blip_model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.blip_model = Blip2(self.model_name, self.model_type,"cpu")


    def load_image(self, leader_image):
        """
        Load an image.

        This method loads an image and set it to self.image.

        Parameters
        ----------
        leader_image : str
            The path to the leader image.

        Returns
        -------
        None
        """
        img = np.load(leader_image)
        self.image = Image.fromarray(img)
        self.visualize_scene()


    def visualize_scene(self):
        """
        Visualize the scene.

        This method uses the blip model to generate a QA response with the question
        "List the objects on the table top". The response is then split into a list of
        objects and stored in self.dialog_dict as a dictionary where the keys are the
        objects and the values are False.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
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
        """
        Get the next object from the dialog dictionary.

        This method iterates through the dialog dictionary and returns the first object
        that has a status of False and is not the current object.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The name of the next object.
        """
        for obj, status in self.dialog_dict.items():
                if status == False and obj != self.current_obj:
                    self.current_obj = obj
                    self.current_obj_status = False
                    break

        return self.current_obj

    
    def  blip_generate_instruction(self, obj):
        """
        Generate an instruction using the Blip model.

        This method generates an instruction from the Blip model by asking the question
        "say something about <obj>" where obj is the object name.

        Parameters
        ----------
        obj : str
            The name of the object.

        Returns
        -------
        str
            The generated instruction.
        """
        text = f"say something about {obj}"
        response = self.blip_model.evaluate_QA(text, self.image)
        return response

    
    def blip_generate_QA(self, text):
        """
        Generate a QA response using the Blip model.

        This method generates a QA response from the Blip model by asking the question
        specified by the input text.

        Parameters
        ----------
        text : str
            The question to ask the Blip model.

        Returns
        -------
        str
            The generated QA response.
        """
        response = self.blip_model.evaluate_QA(text, self.image)
        return response


    def classify_text(self, text):
        """
        Classify a given text using the initialized TextClassifier.

        This method utilizes a pre-trained text classification model to predict
        the label of the input text.

        Parameters
        ----------
        text : str
            The text input to classify.

        Returns
        -------
        str
            The predicted label for the input text.
        """

        self.classifier = TextClassifier(self.label2id)
        resp = self.classifier.inference(text)
        return resp[0]["label"]


    def run(self, text):

        """
        Run the dialog manager.

        This method takes in a text input and classifies it using the initialized
        TextClassifier. Based on the classification result, it generates a response
        using the Blip model or returns the current object name.

        Parameters
        ----------
        text : str
            The text input to classify.

        Returns
        -------
        str
            The generated response.
        """
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
