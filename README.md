# Fake News Detection with NLP and Deep Learning
## Andrea MACHEDA 

This project aims to accurately classify news articles as either real or fake using Natural Language Processing (NLP) and deep learning techniques. The model leverages a dataset of news articles and employs text processing, word embeddings, and an LSTM-based neural network for classification.

## Project Overview

With the growing concern over misinformation, this project provides a solution for detecting fake news by training a binary classification model. The key steps involve:
- Data preprocessing and text cleaning
- Generating word embeddings using Word2Vec
- Building and training an LSTM neural network for classification
- Evaluating the model using accuracy, loss, and a confusion matrix

## Dataset

The project uses two CSV files:
- `Fake.csv`: Contains labeled fake news articles.
- `True.csv`: Contains labeled real news articles.

link dataset : https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Each file has been preprocessed and combined into a single dataset, with an added `label` column to indicate true (1) or fake (0) news.

## Project Structure
- `notebooks/`: Jupyter notebooks with the step-by-step code for data preprocessing, model training, and evaluation.

## Requirements

The project requires Python and the following main libraries:
- `pandas`, `numpy`: For data handling and numerical operations.
- `nltk`: For text processing.
- `gensim`: For Word2Vec model creation.
- `tensorflow` and `keras`: For building the LSTM neural network.
- `sklearn`: For data splitting and confusion matrix generation.
- `matplotlib` and `seaborn`: For plotting training metrics and confusion matrix.
