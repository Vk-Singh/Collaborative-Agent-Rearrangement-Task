import openai
import time
import os
import sys
import gradio as gr
from chat_gpt import *

basepath=os.path.dirname(os.path.dirname(os.path.abspath("")))
if not basepath in sys.path:
    sys.path.append(basepath)


from blip2 import Blip2


INSTRUCTION_GENERATION_PREPROMPT2= \
    "You are a instruction generation leader bot." \
    "A summary of a table-top scene is given starting with \"Summary:\"." \
    "you have to direct me to setup the table top by giving one instruction at a time and responding to my queries." \
    "Identify all the objects from the summary and keep a record of the completed status of every object in the format :" \
    "Name: Cup" \
    "Status: not-placed" \
    "Initial 'Status' of the objects will be 'not-placed'" \
    "If I mention I have completed a step or the object is in place, update the status of the object to 'complete' and generate next instruction." \
    "When the status of all the objects turns to 'complete', reply back by saying 'Thank you, the task is complete.'" \
    "If I ask for the status report, generate the records of all the objects in the above mentioned format."


INSTRUCTION_GENERATION_PREPROMPT1 = \
    "You are a instruction generation leader bot who is playing a Rearrangement game with me. " \
    "A summary of a table-top scene is given starting with Summary:. " \
    "Identify all the objects which are on the table from the summary and keep a record of the status of every object in the format: " \
    "Name: Cup " \
    "Status: not-placed " \
    "Initial 'Status' of the objects will be 'not-placed'." \
    "First you will ask me to start the game. If I am ready then you will direct me to setup the table top by giving one instruction at a time and responding to my queries. " \
    "Generate instruction about a single object at a time only if the status is 'not-placed'." \
    "If I mention I have completed a step or the object is in place, update the status of the object to 'complete' and generate next instruction." \
    "Do not consider table as an object." \
    "Do not ask if I have completed the step every time. If I ask for the next instruction without specifying the step is complete, " \
    "then ask whether the instruction has been completed or not. " \
    "When the status of all the objects turns to 'complete', reply back by saying 'Thank you, the task is complete." \
    "If I ask for the status report, generate the records of all the objects in the above mentioned format." \
    "There is a manager who knows everything about the scene. If you don't know the answer of any query from me, first reason based on the data you have." \
    "If still no able to find an answer, generate a simple question starting with 'BLIP_Ques:' to ask the manager. " \
    "Do not mention that you are replying based on the summary provided." \
    "The game will end when all the objects are placed."


INSTRUCTION_GENERATION_PREPROMPT = \
    "You are a instruction generation leader bot who is playing a Rearrangement game with me." \
    "A summary of a table-top scene is given starting with Summary:. " \
    "Assume all the objects are not on table." \
    "The first step is to identify distinct objects present in the scene and save the status of every object in the format: " \
    "Name: cup " \
    "Color: black " \
    "Status: not-placed " \
    "Initial 'Status' of the objects will be 'not-placed'. " \
    "Treat multiple instances of objects as separate objects. " \
    "If I ask for the status report, generate the records of all the objects in the above mentioned format. " \
    "First ask me to start the game. If I am ready then you will direct me to setup the table top by giving one instruction at a time and responding to my queries. " \
    "Generate instruction about a single object at a time. " \
    "Generate instruction about the object only if the status is 'not-placed'. " \
    "Always start with instructions of plates or saucers on which other objects can be placed." \
    "If I mention I have completed a step or the object is in place, update the status of the object to 'complete'. " \
    "Do not consider table as an object. " \
    "Do not ask if I have completed the step every time. " \
    "There is a manager who knows everything about the scene. If you don't know the answer of any query from me, first reason based on the data you have. " \
    "If not able to find an answer generate a simple question starting with 'BLIP_Ques:' to ask the manager. " \
    "Do not mention that you are replying based on the summary provided. " \
    "Never reply with status of all the objects unless asked." \
    "The game will end when all the objects are placed."
    

INSTRUCTION_GENERATION_PREPROMPT4= \
    "You are a instruction generation leader bot who is playing a Rearrangement Game with me." \
    "A summary of a table-top scene is given starting with Summary:. " \
    "The first step is to identify distinct objects present in the scene." \
    "Treat multiple instances of objects as separate objects. " \
    "Initial 'Status' of the objects will be 'not-placed'. " \
    "Then ask me to start the game. If I am ready then you will direct me to setup the table top by giving one instruction at a time and responding to my queries. " \
    "Generate instruction about a single object at a time. " \
    "Generate instruction about the object only if the object is not placed " \
    "If I mention I have completed a step or the object is in place, update the status of the object to 'complete'. " \
    "When the status of all the objects turns to 'complete', the Rearrganement Game will end." \
    "There is a manager who knows everything about the scene. If you don't know the answer of any query from me, first reason based on the data you have. " \
    "If not able to find an answer generate a simple question starting with 'BLIP_Ques:' to ask the manager. " \
    "Do not mention that you are replying based on the summary provided. " \
    "The game will end when all the objects are placed." \


INSTRUCTION_GENERATION_SUB_PROMPT = \
"ONly after the confirmation from the user that they have completed the task, update the status of the last mentioned object. If all the objects are on the place, end the game or if the user is asking a query, answer it else generate next instruction."

START_PROMPT = 'Ask me to start the rearrangement game.'

LLM_HISTORY = []
USER_HISTORY=[]
SUMMARY = None
TEST_REPLY= None

def filter_gpt_response(resp):
    print(f'resp | {resp}')
    return resp


def get_gpt_response(prompt, filter_func=filter_gpt_response, max_tokens=200, model="gpt-3.5-turbo", temp=0.6):
        forward_flag = False
        retry = 0
        while not forward_flag and retry < 5:
            try:
                response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=temp, max_tokens=max_tokens)
                reply = response['choices'][0]['message']['content']
                reply = filter_func(reply)
                tokens = response['usage']['total_tokens']
                forward_flag = True
                return reply, tokens
                
            except Exception as e:
                print('retrying to call gpt...')
                time.sleep(3)
                retry+=1
        return None


def gen_blip2_prompt(input):
    return f' Question: "{input} Ans:' 

def ask_blip2(question, image):
    resp = manager.vis_model.evaluate_QA(question, image)
    return resp


def generate_gpt_prompt(input, type="chat"):
    prompt = [{"role": "system", "content": INSTRUCTION_GENERATION_PREPROMPT}]
    prompt.append({"role": "system", "content": SUMMARY})

    #print(f'LLM_HIST = {LLM_HISTORY}')
    #print(f'USER HIST = {USER_HISTORY}')
    for q, a in zip(LLM_HISTORY, USER_HISTORY):
        prompt.append({'role': 'user', 'content': a})
        prompt.append({'role': 'assistant', 'content': q})
    prompt.append({'role': 'user', 'content': input})
    prompt.append({"role": "system", "content": INSTRUCTION_GENERATION_SUB_PROMPT})
    return prompt


def my_chatbot(input, history):
    #if not LLM_HISTORY:
     #   setup_gpt_interaction(SUMMARY)
    #print(f"manager {manager.format_out}")
    history = history or []
    my_history = list(sum(history, ()))
    my_history.append(input)
    #my_input = ' '.join(my_history)

    prompt = generate_gpt_prompt(input)
    print(f'prompt : {prompt}')
    output, tokens = get_gpt_response(prompt)
    #print (f"OUTPUT : {output}")
    output = filter_gpt_response(output)
    if "BLIP_Ques1:" in output:
        blip_prompt = gen_blip2_prompt(output)
        blip_output = ask_blip2(blip_prompt,manager.image)
        out = generate_gpt_prompt(blip_output, 'blip')
        print(f'Out_Ques |  {out}')
    print(f'Tokens: {tokens}')
    USER_HISTORY.append(input)
    # Filter GPT Response
    # Call BLIP2 if needed

    LLM_HISTORY.append(output)
    history.append((input, output))
 
    return history, history


def setup_gpt_interaction(summary):
    prompt = generate_gpt_prompt(summary)
    USER_HISTORY.append(summary)
    #print (f'PROMPT INIT : {prompt}')
    reply, tokens = get_gpt_response(prompt, filter_gpt_response)
    print(f'reply: {reply}')
    print(f'tokens: {tokens}')
    LLM_HISTORY.append(reply)
    TEST_REPLY = reply



if __name__ == "__main__":

    key = "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"
    set_api_key(key)
    
    img_path = "src/blip2_agent/images/1.jpg"
    #manager = DescriptionGPT('gpt-3.5-turbo', 'instruct_flant5-xl')
    
    #manager = DescriptionGPT('gpt-3.5-turbo', 'blip2_t5')
    #summary_out,summary_form_out = manager.make_desc(img_path, DESC_PRE_PROMPT)
    
    summary_out1 = "On the counter, there is a square white plate with a glass of water placed on top of it. Next to the glass of water is an oblong red apple. " \
    "The plate does not have any special features or designs. " \
    "There are no other objects present on the counter." \
    " The apple is of oblong shape and the glass of water is oblong as well." \
    " The glass of water is made of glass and is white in color. The plate is of small size."

    summary_out2="The image contains two white plates on the counter. One plate holds" \
    "a glass of water, while the other plate has a red apple. There is also a white" \
    "coffee cup with a lid placed on the counter."

    SUMMARY = 'Summary: ' + summary_out1
    CSS ="""
    .contain { display: flex; flex-direction: column; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; }
    """
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("""<h1><center>Rearrangement Game</center></h1>""")
        chatbot = gr.Chatbot()
        state = gr.State()
        
        txt = gr.Textbox(show_label=False, placeholder="Follow the instructions and if you have any questions, Enter the questions and press enter.").style(container=False)
        txt.submit(my_chatbot, inputs=[txt, state], outputs=[chatbot, state])
        txt.submit(lambda x: gr.update(value=""), None, [txt], queue=False)
    demo.launch(share = False)
