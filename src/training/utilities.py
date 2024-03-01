import evaluate
import numpy as np

from src.data_preparation.prepare import TOKENIZER
from transformers import DataCollatorForTokenClassification

DATA_COLLATOR = DataCollatorForTokenClassification(tokenizer=TOKENIZER)
METRIC = evaluate.load("seqeval")

class LabelList:
    label_names: list = None
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(LabelList, cls).__new__(cls)
        return cls.instance
    
    def __init__(self, label_names: str = None) -> None:
        if label_names:
            self.label_names = label_names

def get_id_and_label_mapping(label_names: list):
    id_to_label = {i: label for i, label in enumerate(label_names)}
    label_to_id = {v: k for k, v in id_to_label.items()}
    return id_to_label, label_to_id

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    label_names = LabelList().label_names

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = METRIC.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }