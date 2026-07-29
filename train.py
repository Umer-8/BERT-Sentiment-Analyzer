import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}


def load_data(path="data/sentiment_dataset.csv"):
    df = pd.read_csv(path)
    df["label"] = df["label"].map(LABEL_MAP)
    return df


def build_datasets(df, test_size=0.2, seed=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
    return train_dataset, test_dataset


def tokenize_datasets(train_dataset, test_dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    return train_dataset, test_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train_model(train_dataset, test_dataset, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer, model


def evaluate_model(trainer):
    return trainer.evaluate()


def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return ID_TO_LABEL[pred]


def save_model(model, tokenizer, path="model"):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def main():
    df = load_data()
    train_dataset, test_dataset = build_datasets(df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, test_dataset = tokenize_datasets(train_dataset, test_dataset, tokenizer)

    trainer, model = train_model(train_dataset, test_dataset, tokenizer)

    metrics = evaluate_model(trainer)
    print("Evaluation metrics:", metrics)

    save_model(model, tokenizer)
    print("Model saved to ./model")

    sample_texts = [
        "This movie was fantastic!",
        "This product is terrible and broke immediately.",
    ]
    for text in sample_texts:
        sentiment = predict_sentiment(text, model, tokenizer)
        print(f"Input: {text}")
        print(f"Predicted Sentiment: {sentiment}\n")


if __name__ == "__main__":
    main()
