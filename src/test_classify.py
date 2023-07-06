import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric


def load_data():
    pass

def read_file(input_path):
    pass


def csv_to_json(input_file):
    pass


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

id2label = {0: "NEXT_STEP", 1: "SUCCESS", -1: "BLIP2"}
label2id = {"NEXT_STEP": 0, "SUCCESS": 1, "BLIP2":-1}

model = AutoModelForSequenceClassification.from_pretrained(

    "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id

)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=,
    eval_dataset=,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
