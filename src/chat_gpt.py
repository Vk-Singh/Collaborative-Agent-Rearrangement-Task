import os
import sys
import numpy as np
import openai
from PIL import Image
import time
import yaml


basepath=os.path.dirname(os.path.dirname(os.path.abspath("")))
if not basepath in sys.path:
    sys.path.append(basepath)



from blip2 import Blip2


def set_api_key(key):
    openai.api_key = key


IMAGE_OBJ ={}
CHATGPT_MODELS = ["gpt_3.5-turbo", "gpt-3.5-turbo-0613"]
BLIP2_MODELS = ["blip2_t5"]
IMAGE_DESC_PROMPT = []



def call_gpt_old(gpt_messages, max_tokens=60, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=0.65, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    tokens = response['usage']['total_tokens']
                
    return reply, tokens

BLIP_PROMPT = 'Describe the image in detail. Do not skip information about any object. Do not hallucinate objects in the image.'


class DialManagerGPT:
    def __init__(self, llm_model="gpt-3.5-turbo",  vision_model="blip2"):
        self.llm_model = llm_model
        self.vis_model_name = vision_model
        self.vis_model = None
        self.image_path = None
        self.image = None
        self.final_description = []
        self.desc_ques = []
        self.desc_ans = []
        self.setup_vision_model()
        self.conversation = []
        self.summary = None

    def make_desc(self, image):
        self.image_path = image
        self._load_image(image)
        count=0
        n_count=20
        for i in range(n_count):
            if not self.desc_ques:
                # list is empty so first message needs to be sent to llm
               vis_ques = BLIP_PROMPT
               self.desc_ques.append(vis_ques)

            else:
                vis_ques = self.generate_blip_prompt(q)
            

            resp = self.vis_model.evaluate_QA(vis_ques, self.image)
            resp = self.clean_blip_response(resp)
           # resp = "test" 
            print (f'BLIP_PROMPT {vis_ques}')
            print('#######################################i')
            print (f'response {resp}')
            print('#######################################')

            self.desc_ans.append(resp[0])
            if count < n_count-1:

                gpt_prompt = self.generate_gpt_prompt("question")
                q, tokens = self.call_gpt(gpt_prompt)
                self.desc_ques.append(q)

                print(f'gpt question = {q}')
                print(f'total tokens = {tokens}')
            count +=1
        
        summary_prompt = self.generate_gpt_prompt("summary")
        self.summary, tokens = self.call_gpt(summary_prompt)
        #print (f'Questions after{self.desc_ques}')
        #print (f'Answers after {self.desc_ans}')
        print(f'Summary: {self.summary}')
        self.generate_conv_output()
	

    def call_gpt(self, prompt, model="gpt-3.5-turbo", temp=0.65, max_tokens=100):
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
        conv = []
        self.conversation.append({'Image': self.image_path})
        self.conversation.append({'GPT_Prompt': QUESTION_PROMPT })
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
        print(f'conversation list: {conv}')
        with open(f'src/blip2_agent/output/conv_{time.strftime("%Y%m%d-%H%M%S")}.yaml', 'w') as f:
            yaml.dump(self.conversation, f)


    def setup_vision_model(self):
        if self.vis_model_name == "blip2":
            self.vis_model = Blip2("blip2_t5", "pretrain_flant5xl", "cpu")


    def _load_image(self, image):
        self.image = Image.open(image).convert("RGB")
        #self.image.show()

  
    def generate_blip_prompt(self, question, len_history=10):
        prompt = ' '
        template = ' Question: {} Answer: {}.'
        ques = self.desc_ques[-len_history:]
        ans = self.desc_ans[-len_history:]
        for i in range(len(ans)):
            prompt = prompt + f'Question: {ques[i]} Answer: {ans[i]}.'
        prompt = prompt + f' Question: "{question} Ans:'      
        return prompt
        

    def generate_gpt_prompt(self, prompt_type="question"):
        prompt = [{"role": "system", "content": QUESTION_PROMPT}]

        for q, a in zip(self.desc_ques, self.desc_ans):
            prompt.append({'role': 'assistant', 'content': 'Question: {}'.format(q)})
            prompt.append({'role': 'user', 'content': 'Answer: {}'.format(a)})
            if prompt_type == "question":
                prompt.append({"role": "system", "content": SUB_QUESTION_PROMPT})
            elif prompt_type == "summary":
                prompt.append({"role": "system", "content": SUMMARY_PROMPT})

        return prompt


    def clean_gpt_response(self, response):
        return response.replace('Question:', '').strip()


    def clean_blip_response(self, response):
        print (f' last ques | {self.desc_ques[-1]}')
        if self.desc_ques[-1] in response:
            return "I don't know. Ask another question."
        else:
            return response


QUESTION_PROMPT = \
        "I have an image of a table top containing objects. " \
        "Ask detailed questions about the objects present on the table. " \
        "Carefully ask me informative and short questions to gather as much information as possible about the objects in the image. " \
        "Answers to these questions will be used to generate instructions to recreate the scene." \
        "Each time ask one question only without giving an answer. " \
        "Try to add variety to the questions and keep the questions short." \
        "I'll reply back starting with \"Ans:\"." \
        "Do not hallucinate information. Only consider information provided by I." \


SUMMARY_PROMPT = \
        "Given above mentioned information, summarize the scene without losing any detail. Do not hallucinate information. The summary can be upto 100 words long."


SUB_QUESTION_PROMPT = \
        "Generate Next Question. Do not ask yes/no questions. \n" \
        "Question: "


QUESTION_INSTRUCTION2 = \
        " It is a normal chat session. Answer basic interaction questions"


GENERATE_INSTRUCTIONS= \
        "Below mentioned is the description of the scene of a table top." \
        "I have all the objects mentioned in the scene. " \
        "Generate step by step detailed instructions to setup the scene."


if __name__ == "__main__":
   # key = "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"
    key = "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"
    
    test = {"role": "user", "content": "The scene contains a mmug on a bowl, a red apple and a glass filled with coffee."}
    set_api_key(key)
    
    messages = [{"role": "system", "content": GENERATE_INSTRUCTIONS}]
    
    messages.append(test)
   # print(messages)
    
    #a, b = call_gpt(messages)
    #print(a)
    #print(f"total tokens = {b}")
    img_path = "src/blip2_agent/images/1.jpg"
    manager = DialManagerGPT()
    manager.make_desc(img_path)
