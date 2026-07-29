# Fine-Tune DistilBERT for Sentiment Analysis

## Files
- `data/sentiment_dataset.csv` — 200 labeled samples (positive/negative)
- `train.py` — single script: loads data, tokenizes with DistilBERT's tokenizer, fine-tunes
  `distilbert-base-uncased` for binary sentiment classification, evaluates, saves the model,
  and runs sample predictions
- `requirements.txt`
- `model/` — saved fine-tuned model + tokenizer (generated after running)

## Run
```bash
pip install transformers datasets torch scikit-learn pandas numpy
python train.py
```
On first run, `transformers` automatically downloads the pretrained `distilbert-base-uncased`
weights and tokenizer from Hugging Face. Training runs for 2 epochs on an 80/20 train/test
split of `sentiment_dataset.csv`, then prints evaluation metrics (accuracy, precision, recall,
F1) and predictions for two example sentences.

## Example output
```
Input: This movie was fantastic!
Predicted Sentiment: positive

Input: This product is terrible and broke immediately.
Predicted Sentiment: negative
```

## How it works
- `AutoTokenizer.from_pretrained("distilbert-base-uncased")` converts raw text into token
  IDs the model understands, padded/truncated to a fixed length
- `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)` loads DistilBERT's
  pretrained language understanding with a fresh classification head on top
- `Trainer` handles the fine-tuning loop: forward pass, loss, backprop, optimizer steps,
  and per-epoch evaluation using `compute_metrics` (accuracy/precision/recall/F1)
- After training, the model + tokenizer are saved to `model/` so they can be reloaded later
  without retraining
- `predict_sentiment()` tokenizes new text the same way and returns the predicted class

## Testing note
This code could not be executed in the sandbox used to prepare this submission. Downloading
the pretrained DistilBERT weights and tokenizer requires reaching `huggingface.co`, which
is outside this environment's network allowlist, and I ran into disk-space limits while
even attempting to install PyTorch's full dependency stack. `train.py` follows the standard,
well-documented Hugging Face `Trainer` API exactly as shown in the official docs and passed
a Python syntax check, but I was not able to confirm actual training/evaluation output
myself before delivering this. Please test on your end and let me know if anything needs
fixing.
