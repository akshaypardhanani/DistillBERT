from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments

from src.data_preparation.load_data import load
from src.data_preparation.prepare import TOKENIZER
from src.training.utilities import DATA_COLLATOR, LabelList, compute_metrics, get_id_and_label_mapping


def get_model_trainer(model_checkpoint: str):
    tokenized_datasets, label_names = load()
    LabelList(label_names)
    id_to_label, label_to_id = get_id_and_label_mapping(label_names)
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    args = TrainingArguments(
        "distilbert-finetuned-ner",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        use_mps_device=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DATA_COLLATOR,
        compute_metrics=compute_metrics,
        tokenizer=TOKENIZER,
    )
    
    return trainer, tokenized_datasets

if __name__ == '__main__':
    trainer, tokenized_dataset = get_model_trainer('distilbert-base-uncased')
    trainer.train()
    metrics = trainer.evaluate(eval_dataset=tokenized_dataset['test'])
    print(f'Evaluated Metrtics Are: {metrics}')
    trainer.save_model('distilbert-finer-uncased')
    