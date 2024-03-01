from src.data_preparation.prepare import TOKENIZER, _filter_dataset, _load_dataset, tokenize_and_align_labels


def load(labels_list: list = [138, 139, 42, 43]):
    dataset, label_names = _load_dataset("nlpaueb/finer-139")
    dataset = _filter_dataset(dataset, labels_list)
    dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=['id', 'tokens', 'ner_tags']
    )
    
    return dataset, label_names
