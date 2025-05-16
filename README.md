# Sentiment-Analysis-with-BERT-on-IMDb-Dataset

This project fine-tunes a **BERT-based transformer model** for binary sentiment classification of IMDb movie reviews (positive or negative). Developed for COMP1818 coursework, it showcases advanced NLP using Hugging Face Transformers, PyTorch, and GPU acceleration. The model achieves **92.86% accuracy** and **92.98% F1-score**.

---

## Key Features

- Fine-tunes `bert-base-uncased` for binary classification
- Preprocessing, tokenization, and model training with Hugging Face `Trainer`
- Evaluation using Accuracy, Precision, Recall, and F1-score
- Inference script for new custom review predictions
- Handles sarcasm and nuance analysis in discussion

---

## Dataset

This project uses the [IMDb Movie Review Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### How to Get It

1. Download the dataset CSV file from:  
    https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Rename it to: `IMDB Dataset.csv`
3. Place it inside the `data/` folder
