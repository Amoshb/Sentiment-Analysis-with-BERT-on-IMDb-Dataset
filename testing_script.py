from transformers import pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_sentiment(texts, model_dir):
    """
    Predict sentiment for a list of texts using a pre-trained model.
    
    Args:
        texts (list of str): Input texts to classify.
        model_dir (str): Directory containing the saved model and tokenizer.
    
    Returns:
        list of dict: Predictions with labels and scores.
    """
    # Load the classification pipeline
    classifier = pipeline("text-classification", model=model_dir, tokenizer=model_dir, device=0 if device == "cuda" else -1)

    # Map Hugging Face labels to user-friendly labels
    label_mapping = {"LABEL_0": "negative", "LABEL_1": "positive"}

    # Perform predictions and replace labels with user-friendly ones
    results = classifier(texts)
    for result in results:
        result['label'] = label_mapping[result['label']]  # Replace LABEL_0 and LABEL_1
    return results


# Example Usage
if __name__ == "__main__":
    # Define the model directory (update if different)
    model_directory = "./saved_model"

    # Example texts for testing
    sample_texts = [
        "The movie was absolutely fantastic!",
        "The plot was boring and uninspired.",
        "I loved the acting and direction.",
        "This is one of the worst movies I've seen.",
        "The movie was good but can get better.",
        "The movie had a great cast but ultimately failed to deliver.",
        "I really loved watching paint dry more than this movie."

    ]

    # Get predictions
    predictions = predict_sentiment(sample_texts, model_directory)

    # Print predictions
    print("Predictions:")
    for i, prediction in enumerate(predictions):
        print(f"Text: {sample_texts[i]}")
        print(f"Prediction: {prediction['label']}, Confidence: {prediction['score']:.2f}\n")
