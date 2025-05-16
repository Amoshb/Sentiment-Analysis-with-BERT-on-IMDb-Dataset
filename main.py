import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Ensure GPU is used if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: Load Dataset
def load_and_preprocess_data(path):
    data = pd.read_csv(path)  # Replace with your IMDb dataset path
    data['label'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    return data

dataset_path = 'IMDB Dataset.csv'
data = load_and_preprocess_data(dataset_path)

# Step 2: Split Dataset into Train and Test
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Step 3: Tokenization
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(batch):
    return tokenizer(batch['review'], padding="max_length", truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns and set format
train_dataset = train_dataset.remove_columns(['review', 'sentiment'])
test_dataset = test_dataset.remove_columns(['review', 'sentiment'])
train_dataset.set_format(type='torch')
test_dataset.set_format(type='torch')

# Step 4: Load Pre-trained Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Step 5: Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none" 
)

# Step 6: Metrics for Evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Step 7: Trainer Initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Step 8: Train the Model
trainer.train()

# Step 9: Evaluate the Model
metrics = trainer.evaluate()
print("Evaluation Results:", metrics)

# Step 10: Save Model and Tokenizer
model_dir = "./saved_model"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print(f"Model and tokenizer saved to {model_dir}")

# Step 11: Inference on New Data
def predict_sentiment(texts, model_dir):
    from transformers import pipeline

    classifier = pipeline("text-classification", model=model_dir, tokenizer=model_dir, device=0 if device == "cuda" else -1)
    label_mapping = {"LABEL_0": "negative", "LABEL_1": "positive"}
    results = classifier(texts)
    for result in results:
        result['label'] = label_mapping[result['label']]  # Replace LABEL_0 and LABEL_1
    return results

# Example Inference
sample_texts = ["The movie was absolutely fantastic!", "The plot was boring and uninspired."]
predictions = predict_sentiment(sample_texts, model_dir)
print("Predictions:", predictions)

