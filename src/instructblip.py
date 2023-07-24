from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image


#model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
#processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")


################################### BLIP2 ###############
#model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl')
#processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')
#device = "cuda" if torch.cuda.is_available() else "cpu"


device = "cpu"
model.to(device)

image = Image.open('src/blip2_agent/images/2.jpg').convert("RGB")
prompt = "Is there a knife on the table?"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

outputs = model.generate(**inputs,do_sample=False,num_beams=5,max_length=256,min_length=1,top_p=0.9,repetition_penalty=1.5,length_penalty=1.0,temperature=1,)

generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
