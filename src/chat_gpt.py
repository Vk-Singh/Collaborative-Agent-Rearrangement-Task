import os
import sys
import numpy as np
import openai
from PIL import Image
import time
import yaml
import pickle


basepath=os.path.dirname(os.path.dirname(os.path.abspath("")))
if not basepath in sys.path:
    sys.path.append(basepath)


from blip2 import Blip2
from InstructBlip2 import InstructBlip


def set_api_key(key):
    """
    Set the OpenAI API key.

    Parameters
    ----------
    key : str
        The OpenAI API key.

    Returns
    -------
    None
    """
    openai.api_key = key


CHATGPT_MODELS = ["gpt_3.5-turbo", "gpt-3.5-turbo-0613"]
BLIP2_MODELS = ["blip2_t5", "blip2_opt"]

BLIP_PROMPT = 'Describe the image in detail. Do not skip information about any object. Do not hallucinate objects and information in the image.'


class DescriptionGPT:
    def __init__(self, llm_model="gpt-3.5-turbo",  vision_model="blip2_t5"):
        """
        Initialize the DescriptionGPT object.

        Parameters
        ----------
        llm_model : str
            The LLM model to use. Defaults to "gpt-3.5-turbo".
        vision_model : str
            The vision model to use. Defaults to "blip2_t5".

        Returns
        -------
        None
        """
        self.llm_model = llm_model
        self.vis_model_name = vision_model
        self.blip_hist_len = 0
        self.vis_model = None
        self.image_path = None
        self.image = None
        self.final_description = []
        self.desc_ques = []
        self.desc_ans = []
        self.conversation = []
        self.summary = None
        self.vis_model_type = None
        self.format_out=None
        self.DESC_PRE_PROMPT=None
        self.setup_vision_model()


    def make_desc(self, image, DESC_PRE_PROMPT, num_count=10):
        """
        Generate a description of an image given a prompt and a number of count.

        Parameters
        ----------
        image : str
            The path to the image file.
        DESC_PRE_PROMPT : str
            The prompt to use for generating the description.
        num_count : int
            The number of times to generate a description. Defaults to 10.

        Returns
        -------
        tuple
            A tuple containing the generated summary and the generated description.
        """
        self.DESC_PRE_PROMPT=DESC_PRE_PROMPT
        self.image_path = image
        self._load_image(image)
        count=0
        n_count=num_count
        #########TESTING CAPTION################
        print(f'Caption : {self.vis_model.evaluate_caption(self.image)}')
        for i in range(n_count):
            if not self.desc_ques:
                # list is empty so first message needs to be sent to llm
               vis_ques = BLIP_PROMPT
               self.desc_ques.append(vis_ques)

            else:
                vis_ques = self.generate_blip_prompt(q, self.blip_hist_len)

            resp = self.vis_model.evaluate_QA(vis_ques, self.image)
            resp = self.clean_blip_response(resp)
           # resp = "test"
            print (f'BLIP_PROMPT {vis_ques}')
            print('######################################')
            print (f'response {resp}')
            print('######################################')
            if isinstance(resp, str):
                self.desc_ans.append(resp)
            elif isinstance(resp, list):
                self.desc_ans.append(resp[0])

            if count < n_count-1:

                gpt_prompt = self.generate_gpt_prompt("question")
                q, tokens = self.call_gpt(gpt_prompt)
                self.desc_ques.append(q)
                print(f'gpt question = {q}')
                print(f'total tokens = {tokens}')
            count +=1
        out_pr = self.generate_gpt_prompt("output")
        self.format_out, tokens = self.call_gpt(out_pr, 300)
        print(f'output = {self.format_out}')
        summary_prompt = self.generate_gpt_prompt("summary")
        self.summary, tokens = self.call_gpt(summary_prompt, 300)
        #print (f'Questions after{self.desc_ques}')
        #print (f'Answers after {self.desc_ans}')
        print(f'Summary: {self.summary}')
        self.generate_conv_output()
        return (self.summary, self.format_out)
	

    def call_gpt(self, prompt,max_tokens=100, model="gpt-3.5-turbo", temp=0.6):
        """
        Calls the GPT model to generate text based on the given prompt.

        Parameters
        ----------
        prompt : list
            A list of dictionaries containing the prompt to send to the GPT model.
        max_tokens : int, optional
            The maximum number of tokens to generate. Defaults to 100.
        model : str, optional
            The name of the GPT model to use. Defaults to "gpt-3.5-turbo".
        temp : float, optional
            The temperature parameter to use when generating text. Defaults to 0.6.

        Returns
        -------
        tuple
            A tuple containing the generated text and the number of tokens used.
        """
        forward_flag = False
        retry = 0
        while not forward_flag and retry < 5:
            try:
                response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=temp, max_tokens=max_tokens)
                reply = response['choices'][0]['message']['content']
                reply = self.clean_gpt_response(reply)
                tokens = response['usage']['total_tokens']
                forward_flag = True
                return reply, tokens
                
            except Exception as e:
                print('retrying to call gpt...')
                time.sleep(3)
                retry+=1
        return None


    def generate_conv_output(self):
        """
        Generate the conversation output into a yaml file and a pickle file.
        
        The output yaml file will contain the following information:
        - The image path
        - The BLIP2 model name and type
        - The length of the BLIP history
        - The GPT prompt
        - The GPT sub prompt
        - The BLIP prompt
        - The summary prompt
        - The conversation between the user and the BLIP2 model
        - The summary generated by the GPT model
        - The final output of the conversation
        
        The output pickle file will contain the following information:
        - The list of questions asked by the user
        - The list of answers given by the BLIP2 model
        """
        conv = []
        self.conversation.append({'Image': self.image_path})
        self.conversation.append({'BLIP2 Model' : f'Model : {self.vis_model_name} | Model type : {self.vis_model_type}'})
        self.conversation.append({'BLIP HISTORY LENGTH': self.blip_hist_len})
        self.conversation.append({'GPT_Prompt': self.DESC_PRE_PROMPT })
        self.conversation.append({'GPT_SUB_PROMPT': SUB_QUESTION_PROMPT})
        self.conversation.append({'Blip_Prompt': BLIP_PROMPT})
        self.conversation.append({'Summary_Prompt': SUMMARY_PROMPT})
        count=0
        for q,a in zip(self.desc_ques, self.desc_ans):
            if count ==0:
                conv.append(f'BLIP_PROMPT_QUESTION: {q}')
                count +=1
            else:
                conv.append(f'GPT_QUESTION: {q}')
            conv.append(f'BLIP_ANSWER: {a}')
        self.conversation.append({'CONVERSATION': conv})
        self.conversation.append({'SUMMARY': self.summary})
        self.conversation.append({'OUTPUT': self.format_out})

        #print(f'conversation list: {conv}')
        curr_time = time.strftime("%Y%m%d-%H%M%S")
        with open(f'src/blip2_agent/output/conv_{curr_time}.yaml', 'w') as f:
            yaml.dump(self.conversation, f)

        with open(f'src/blip2_agent/output/pickle/conv_{curr_time}.pkl', 'wb') as f:
            pickle.dump((self.desc_ques, self.desc_ans), f)


    def setup_vision_model(self):
        """
        Initialize the vision model based on the specified model name.

        This method sets up the vision model (`self.vis_model`) and its corresponding
        type (`self.vis_model_type`) based on the value of `self.vis_model_name`.

        Supported `vis_model_name` values:
        - "instruct_vicuna-7b": Initializes an InstructBlip model with 'Vicuna-7b'.
        - "instruct_flant5-xl": Initializes an InstructBlip model with 'flant5-xl'.
        - "blip2_opt": Initializes a Blip2 model with 'blip2_opt' and 'pretrain_opt2.7b'.
        - "blip2_t5": Initializes a Blip2 model with 'blip2_t5' and 'pretrain_flant5xl'.

        The vision model is configured to run on the CPU.
        """

        if self.vis_model_name == "instruct_vicuna-7b":
            self.vis_model =InstructBlip('Vicuna-7b')
            self.vis_model_type = "InstructBlip"

        elif self.vis_model_name == "instruct_flant5-xl":
            self.vis_model = InstructBlip('flant5-xl')
            self.vis_model_type = "InstructBlip"

        elif self.vis_model_name == "blip2_opt":
            self.vis_model = Blip2("blip2_opt", "pretrain_opt2.7b", "cpu")
            self.vis_model_type = 'pretrain_opt2.7b'

        elif self.vis_model_name == "blip2_t5":
            self.vis_model = Blip2("blip2_t5", "pretrain_flant5xl", "cpu")
            self.vis_model_type = 'pretrain_flat5xl'


    def _load_image(self, image):
        """
        Load an image from a specified path and convert it to RGB format.

        Parameters
        ----------
        image : str
            The path to the image file to be loaded.

        Returns
        -------
        None
        """

        self.image = Image.open(image).convert("RGB")
        #self.image.show()

  
    def generate_blip_prompt(self, question, len_history=10):
        """
        Generate a BLIP prompt using the latest question and a specified number
        of previous question-answer pairs from the description history.

        Parameters
        ----------
        question : str
            The current question to be added to the prompt.
        len_history : int, optional
            The number of previous question-answer pairs to include in the prompt.
            Defaults to 10.

        Returns
        -------
        str
            A formatted prompt string containing the specified number of 
            historical question-answer pairs followed by the current question.
        """

        prompt = ' '
        if len_history > 0:
            ques = self.desc_ques[:-1]
            ques = ques[-len_history:]
            ans = self.desc_ans[-len_history:]
            print(f"len ques {len(ques)}")
            print (f"len ans {len(ans)}")
            assert len(ques) == len(ans)
            for i in range(len(ans)):
                prompt = prompt + f' Question: {ques[i]} Answer: {ans[i]}.'
        prompt = prompt + f' Question: "{question} Ans:'      
        return prompt
        

    def generate_gpt_prompt(self, prompt_type="question"):
        """
        Generate a GPT prompt using the latest question and a specified number
        of previous question-answer pairs from the description history.

        Parameters
        ----------
        prompt_type : str, optional
            The type of prompt to be generated. Can be one of "question", "summary", or "output".
            Defaults to "question".

        Returns
        -------
        list
            A list of dictionaries containing the formatted prompt string with the specified
            number of historical question-answer pairs followed by the current question.
        """
        prompt = [{"role": "system", "content": self.DESC_PRE_PROMPT}]

        for q, a in zip(self.desc_ques, self.desc_ans):
            prompt.append({'role': 'assistant', 'content': 'Question: {}'.format(q)})
            prompt.append({'role': 'user', 'content': 'Answer: {}'.format(a)})
        if prompt_type == "question":
            prompt.append({"role": "system", "content": SUB_QUESTION_PROMPT})
        elif prompt_type == "summary":
            prompt.append({"role": "system", "content": SUMMARY_PROMPT})
        elif prompt_type =="output":
            prompt.append({"role": "system", "content": OUTPUT_GENERATION})
        return prompt


    def clean_gpt_response(self, response):
        """
        Clean a response from GPT by removing the "Question:" string and stripping any leading/trailing whitespace.

        Parameters
        ----------
        response : str
            The response from GPT to be cleaned.

        Returns
        -------
        str
            The cleaned response string.
        """
        return response.replace('Question:', '').strip()


    def clean_blip_response(self, response):
        """
        Clean a response from BLIP by checking if the last question asked is present in the response.
        If the last question is present, return a string indicating that the bot does not know the answer.
        Otherwise, return the response unchanged.
        """
        print (f' last ques | {self.desc_ques[-1]}')
        if self.desc_ques[-1] in response:
            return "I don't know. Ask another question."
        else:
            return response


DESC_PRE_PROMPT = \
        "You are a robot for summarizing a table top image containing objects." \
        "You need to gather information about the objects." \
        "You can gather the information by asking questions to me."  \
        "Ask simple questions about the objects present on the table." \
        "Each time you can ask only one question without giving the answer." \
        "Make sure that only a single object can be on a single plate. If there are multiple objects on a plate, verify the number of plates/bowl." \
        "Save the information about each object in the format: " \
        "Name: Cup" \
        "Color: Red" \
        "Placed on: Plate" \
        "Special Feature: Color printing on the cup"\
        \
        "There can be multiple objects of the same type. In that case, add suffix to the 'Name' of the object and save information as:" \
        "Name: Cup2" \
        "Color: Blue" \
        "Placed on: Table"\
        "Special Feature: broken cup" \
        "I'll reply back starting with Ans:." \
        \
        "Do not hallucinate information. Only consider information provided by I." \
        "As you start getting information about objects, use those information to ask meaningful questions. For eg: after getting color of an object as 'red' use the color 'red' in the questions to emphasize the object." \
        "When I specifically say 'Give me image output', generate the information in above mentioned format."


DESC_PRE_PROMPT2 = \
        "I have an image of a table top containing objects. " \
        "Ask detailed questions about the objects present on the table. " \
        "Do not give options while asking questions." \
        "Answers to these questions will be used to generate instructions to recreate the scene." \
        "Focus on color , count and shape of the objects." \
        "Each time ask one question only without giving an answer. " \
        "Try to add variety to the questions and keep the questions short." \
        "I'll reply back starting with \"Ans:\"." \
        "Do not hallucinate information. Only consider information provided by I." \


SUMMARY_PROMPT = \
        "Given above mentioned information, summarize the contents of the image. Do not give conflicting information. Do not hallucinate information. If it says 'on top of' while mentioning position, that means 'furthur up on the table'. The summary can be upto 100 words long."


SUB_QUESTION_PROMPT = \
        "Generate Next Question. Do not ask yes/no questions. Do not ask about more than one object in a single question. \n" \
        "Question: "


OUTPUT_GENERATION = \
        "Give me image output."


GENERATE_INSTRUCTIONS= \
        "Below mentioned is the description of the scene of a table top." \
        "I have all the objects mentioned in the scene. " \
        "Generate step by step detailed instructions to setup the scene."


if __name__ == "__main__":
    key = "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"
    set_api_key(key)
    img_path = "src/blip2_agent/images/4.jpg"
    #manager = DescriptionGPT('gpt-3.5-turbo', 'instruct_flant5-xl')
    manager = DescriptionGPT('gpt-3.5-turbo', 'blip2_t5')
    manager.make_desc(img_path, DESC_PRE_PROMPT)
