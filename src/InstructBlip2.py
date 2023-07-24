
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch


INSTRUCT_BLIP_DICT = {
    'Vicuna-7b' : 'Salesforce/instructblip-vicuna-7b',
    'flant5-xl' : 'Salesforce/instructblip-flan-t5-xl'
}

class InstructBlip():
    def __init__(self, model, device = "cpu", bit8=True):
        self.tag = model
        self.bit8 = bit8
        self.device = device
        dtype = {'load_in_8bit': True} if self.bit8 else {'torch_dtype': torch.float16}
        self.instruct_blip_processor = InstructBlipProcessor.from_pretrained(INSTRUCT_BLIP_DICT[self.tag])
        self.instruct_blip_model = InstructBlipForConditionalGeneration.from_pretrained(INSTRUCT_BLIP_DICT[self.tag])
        

    def evaluate_QA(self, question, raw_image):
        #print(f'question = {question}')
        #print(f'type question = {type(question)}')
        inputs = self.instruct_blip_processor(raw_image, question, return_tensors="pt").to(self.device)
        out = self.instruct_blip_model.generate(**inputs, do_sample=False,num_beams=5, max_length=256, min_length=1, \
                                                top_p=0.9,repetition_penalty=1.5,length_penalty=1.0, \
                                                temperature=0.65,)
        answer = self.instruct_blip_processor.decode(out[0], skip_special_tokens=True)
        return answer


    def evaluate_caption(self, raw_image):
        caption = self.evaluate_QA(raw_image, 'a photo of')
        caption = caption.replace('\n', ' ').strip()
        return caption
        

if __name__ == "__main__":
    tag = "flant5-xl"

    b = INSTRUCT_BLIP_DICT(tag)
