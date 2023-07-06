import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline
import numpy as np

#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#model = AutoModelForSequenceClassification.from_pretrained("models")
text = "explain in a fruitful way the shade of the apple?"

#encoding = tokenizer(text, return_tensors="pt")

clf = pipeline("text-classification", "models/classifier")
#encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

#outputs = trainer.model(**encoding)

print(clf(text))
