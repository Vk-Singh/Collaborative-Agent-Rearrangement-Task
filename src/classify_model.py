import pandas as pd
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from transformers import DistilBertTokenizerFast
import torch
from transformers import pipeline


class SetDataset(torch.utils.data.Dataset):
    def __init__(self, text, labels, tokenizer):
        """
        Args:
            text (List[str]): List of strings to be encoded.
            labels (List[int]): List of labels corresponding to the text.
            tokenizer (transformers.PreTrainedTokenizerFast): Tokenizer to use for encoding the text.
        """
        self.encodings = tokenizer(text, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieves a single data point from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            dict: A dictionary containing the encoded text tensors and the corresponding label tensor.
                Keys include all those present in the encodings and 'labels'.
        """

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.labels)


class TextClassifier:

    def __init__(self, label2id):
        """
        Initialize a TextClassifier with a label-to-id mapping.

        Args:
            label2id (dict): A dictionary mapping labels to IDs.
        """
        self.label2id = label2id

    def train(self, train_dataset, save_path="src/blip2_agent/models/classifier", labels=3):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=labels)
        model.to(device)
        model.train()

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        optim = AdamW(model.parameters(), lr=5e-5)

        for epoch in range(3):
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optim.step()
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)

    def inference(self, text, model='src/blip2_agent/models/classifier'):

        """
        Perform text classification inference using a pre-trained model.

        Args:
            text (str): The text to classify.
            model (str, optional): The path to the pre-trained model. Defaults to 'src/blip2_agent/models/classifier'.

        Returns:
            dict: A dictionary containing the predicted label and the corresponding score.
        """
        clf = pipeline("text-classification", model)
        return clf(text)


if __name__ == "__main__":

    label2id = {"NEXT_STEP": 0, "SUCCESS": 1, "BLIP":2}
    train_texts = df['text'].to_list()
    train_labels = df['label'].to_list()
    train_labels = [label2id[x] for x in train_labels]
    df = pd.read_csv('classifier.csv',header=0)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_dataset = SetDataset(train_texts, train_labels, tokenizer)
