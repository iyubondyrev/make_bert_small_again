from seqeval.metrics import f1_score
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
import torch
from dataset import FilteredDataCollatorForTokenClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_f1(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    for preds, labs in zip(predictions, labels):
        assert len(preds) == len(labs)
    
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    f_score= f1_score(y_true=true_labels, y_pred=true_predictions)
    return {
        "f1": f_score
    }


def compute_f1_on_dataset(model: AutoModelForTokenClassification, tokenized_dataset: Dataset, tokenizer: AutoTokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data_collator = FilteredDataCollatorForTokenClassification(tokenizer)
    validation_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=16,
        collate_fn=data_collator,
    )

    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    all_predictions = []
    all_labels = []

    for batch in tqdm(validation_dataloader):
        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
        labels = batch["labels"].to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = labels.cpu().numpy()

        for preds, labs in zip(predictions, labels):
            valid_preds = [label_names[p] for p, l in zip(preds, labs) if l != -100]
            valid_labels = [label_names[l] for l in labs if l != -100]
            all_predictions.append(valid_preds)
            all_labels.append(valid_labels)

    return f1_score(y_true=all_labels, y_pred=all_predictions)