# Multilingual Spam Classification

## Overview

This project focuses on building a multilingual spam classification system using natural language processing (NLP) techniques. The goal is to develop a machine learning pipeline that can accurately classify messages as either spam or ham (non-spam) across multiple languages.

## Project Structure

The project is organized into several main steps, each serving a specific purpose:

### 1. Data Importing and Cleaning

- Utilizes Google Colab for data exploration and analysis.
- Reads the dataset from a CSV file (`spam.csv`) using pandas.
- Focuses on the 'v1' (Label) and 'v2' (Message) columns.
- Handles any null values in the dataset.

### 2. Exploring the Data

- Displays the first few rows of the dataset to get an initial understanding.
- Identifies unique labels in the 'Label' column (ham, spam).
- Checks for null values in the dataset.
- Visualizes the distribution of spam and ham messages using a count plot.

### 3. Preprocessing the Data

- Defines a preprocessing function to clean and standardize the text data.
- Applies the preprocessing function to the 'Message' column.
- Encodes labels (0 for ham, 1 for spam) using LabelEncoder from scikit-learn.
- Visualizes the distribution of encoded labels.

### 4. Converting Messages to Embeddings

- Utilizes the SentenceTransformer library for converting text messages to embeddings.
- Installs the required library and loads a pre-trained multilingual model (`distiluse-base-multilingual-cased`).
- Encodes the messages into numerical vectors using the loaded model.

### 5. Pipelining

- Defines a pipeline for the entire process, including data importing, cleaning, preprocessing, and embeddings.
- Uses TPOT (Tree-based Pipeline Optimization Tool) to automatically search for the best machine learning pipeline for the task.

## Instructions for Running the Notebook

1. Make sure to mount Google Drive using the provided code.
2. Update the file path to the dataset if needed.
3. Follow the notebook sequentially, running each code cell.

## Dependencies

- pandas
- seaborn
- re
- matplotlib
- scikit-learn
- sentence-transformers
- TPOT

Install the required dependencies using the following:

```python
%pip install pandas seaborn matplotlib scikit-learn sentence-transformers TPOT
