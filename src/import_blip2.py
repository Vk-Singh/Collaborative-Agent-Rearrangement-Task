import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


BLIP2DICT = {
    'FlanT5 XXL': 'Salesforce/blip2-flan-t5-xxl',
    'FlanT5_XL_COCO': 'Salesforce/blip2-flan-t5-xl-coco',
    'OPT6.7B COCO': 'Salesforce/blip2-opt-6.7b-coco',
    'OPT2.7B COCO': 'Salesforce/blip2-opt-2.7b-coco',
    'FlanT5 XL': 'Salesforce/blip2-flan-t5-xl',
    'OPT6.7B': 'Salesforce/blip2-opt-6.7b',
    'OPT2.7B': 'Salesforce/blip2-opt-2.7b',
}



class Blip2():
    def __init__(self, model, device_id=0, bit8=True):
        # load BLIP-2 to a single gpu
        """
        Initialize a Blip2 model.

        Parameters
        ----------
        model : str
            The name of the model.
        device_id : int, optional
            The ID of the GPU to use. Defaults to 0.
        bit8 : bool, optional
            Whether to load the model in 8-bit mode. Defaults to True.

        Attributes
        ----------
        tag : str
            The name of the model.
        bit8 : bool
            Whether the model is loaded in 8-bit mode.
        device_gpu : str
            The device name for the GPU.
        device : str
            The device name for the CPU.
        blip2_processor : Blip2Processor
            The preprocessor for the Blip2 model.
        blip2 : Blip2ForConditionalGeneration
            The Blip2 model itself.
        """
        self.tag = model
        self.bit8 = bit8
        self.device_gpu = 'cuda:{}'.format(device_id)
        self.device = "cpu"
        dtype = {'load_in_8bit': True} if self.bit8 else {'torch_dtype': torch.float16}
        self.blip2_processor = Blip2Processor.from_pretrained(BLIP2DICT[self.tag])
        self.blip2 = Blip2ForConditionalGeneration.from_pretrained(BLIP2DICT[self.tag])
        
    def ask(self, raw_image, question):
        inputs = self.blip2_processor(raw_image, question, return_tensors="pt").to(self.device, torch.float16)
        out = self.blip2.generate(**inputs)
        answer = self.blip2_processor.decode(out[0], skip_special_tokens=True)
        return answer

if __name__ == "__main__":
    tag = "FlanT5_XL_COCO"

    b = Blip2(tag)
