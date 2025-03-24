import time
import openai
import pickle


openai.api_key = "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"

SUMMARY_PROMPT = \
        "Given above mentioned conversation, summarize the contents of the image and create a description. Do not give conflictory information like 'utensil is empty' and 'the same utensil has a cup on it'. A different utensil can have objects on it" \


CONFLICT_PROMPT = \
        "Find conflicting information in above mentioned conversation."

QUES_LIST = []
ANS_LIST = []

def call_gpt(gpt_msg, max_tokens=1000, model = "gpt-3.5-turbo-0613"):
    """
    Calls the GPT model to generate text based on the given prompt.

    Parameters
    ----------
    gpt_msg : list
        A list of dictionaries containing the prompt to send to the GPT model.
    max_tokens : int, optional
        The maximum number of tokens to generate. Defaults to 1000.
    model : str, optional
        The name of the GPT model to use. Defaults to "gpt-3.5-turbo-0613".

    Returns
    -------
    tuple
        A tuple containing the generated text and the number of tokens used.
    """
    response = openai.ChatCompletion.create(model=model, messages=gpt_msg, temperature=0.65, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    tokens = response['usage']['total_tokens']
    return reply, tokens



if __name__ == "__main__":
    QUES_LIST = ['Describe the image in detail. Do not skip information about any object. Do not hallucinate objects in the image.','What color is the plate?','How many apples are on the table?','What shape is the glass of water?','Is there anything else on the table besides the apple, glass of water, and plate?', 'Is the apple whole or sliced?','What is the size of the plate?', 'Is the glass of water full or partially filled?', 'What is the material of the plate and the glass?', 'Is the apple red or green in color?', 'Is the glass of water sitting directly on the table or is it on a coaster or placemat?','Is there any cutlery present on the table?', 'Is the apple stem intact or removed?', 'Is the plate empty or does it have food on it?', 'Are there any napkins or tissues on the table?', 'Are there any decorations or ornaments on the table?', 'Is the table made of wood or another material?', 'Is the glass of water positioned towards the center of the table or towards the edge?', 'Is the apple placed on the plate or directly on the table?', 'Is the plate empty or does it have any residue or crumbs on it?' ]
    ANS_LIST = ['There is an apple, a glass of water, and a plate on the counter','white', 'one', 'oblong', 'no', 'whole', 'the plate is the same size as the glass of water', 'full', 'the plate and the glass are made of glass', 'red', 'the glass of water is sitting directly on the table', 'no', 'intact', 'empty', 'no', 'no', 'wood', 'the glass of water is positioned towards the center of the table', 'the apple is placed on the plate', 'empty' ]

    assert len(QUES_LIST) == len(ANS_LIST)

    #prompt = [{"role": "system", "content": QUESTION_PROMPT}]
    prompt = []
    for q,a in zip(QUES_LIST, ANS_LIST):
        prompt.append({'role': 'assistant', 'content': 'Question: {}'.format(q)})
        prompt.append({'role': 'user', 'content': 'Answer: {}'.format(a)})
    prompt.append({"role": "system", "content": SUMMARY_PROMPT})

    reply, tokens = call_gpt(prompt)

    prompt2 = []
    prompt2.append({'role': 'user', 'content': reply})
    prompt2.append({'role': 'system','content': CONFLICT_PROMPT})
    r, t = call_gpt(prompt2)
    print (f'Conflict Prompt reply : {r}')

"""
    with open('src/blip2_agent/output/pickle/conv_20230707-101833.pkl', 'rb') as f:
        q, a = pickle.load(f)

    print(a)
"""
