# DistillBERT
Fine Tuning DistillBERT on the FiNER-139 dataset

### Model
#### Checkppoints
The model Checkpoints are in the `distilbert-finetuned-ner` directory at the root. `checkpoint-1407` is the one on which all the evaluation has been done

#### Saved Model
This is in the `distilber-finer-tuned` directory.
TODO: To be published to HuggingFace

#### ONNX runtime
See TODOs


#### Directory Structure
```
----DISTILLBERT
  |_ distilbert-finetuned-ner
  |_ src
  |  |_ data_preparation
  |  |_ training
  |_ DatExploration.ipynb
```

### Data Exploration
The file in which we examine the dataset and see the distribution of the tokens and of the labels is `DatExploration.ipynb`
The 4 labels chosen to evaluate on are:
* B-ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1
* I-ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1
* B-DebtInstrumentMaturityDate
* I-DebtInstrumentMaturityDate

#### The obtained metrics on model evaluation are:
| eval_loss  | eval_precision  |  eval_recall |  eval_f1 |  eval_accuracy |
|---|---|---|---|---|
| 0.044684 | 0.770968 | 0.784893 | 0.777868  | 0.976305 |

#### Dependencies
All the dependencies are defined in `requirements.txt` They should be installed in a new venv by running `python -m pip install -r requirements.txt` from the repo root.


#### TODOs
1. Export to ONNX 
2. Evaluate performance on ONNX runtime compared to the original Distil-BERT model
3. Write to Hugging Face