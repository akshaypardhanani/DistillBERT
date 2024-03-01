from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


TOKENIZER = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def _load_dataset(dataset_name: str):
    dataset = load_dataset(dataset_name)
    label_names = dataset["train"].features["ner_tags"].feature.names
    return dataset, label_names
    

def _filter_dataset(dataset: Dataset, tags: list):
    filtered_dataset = dataset.filter(lambda example: len(set(example['ner_tags'])) > 1)
    filtered_dataset = filtered_dataset.filter(lambda example: any(t in example['ner_tags'] for t in tags))
    return filtered_dataset


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = TOKENIZER(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
