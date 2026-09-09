# Text Analyzer and Classification System

## Project Description

This project is a Machine Learning based Text Analyzer that classifies text into two categories:

- Technology
- Sports

The project uses TF-IDF for converting text into numerical features and Logistic Regression for classification.

It also provides file analysis features such as:

- Text classification
- File statistics
- Word frequency
- Category summary
- Model accuracy
- Confidence score

## Technologies Used

- Python
- Scikit-learn
- TF-IDF
- Logistic Regression
- Counter
- Machine Learning
- Natural Language Processing

## Project Structure

Text-Analyzer/
│
├── classifier.py
├── test_model.py
├── accuracy_test.py
├── analyzer.py
├── tfidf_analyzer.py
├── sample.txt
├── README.md
│
└── data/
    ├── tech.txt
    └── sports.txt

## How the Project Works

### Step 1: Training Data

The project reads training sentences from:

- data/tech.txt
- data/sports.txt

### Step 2: TF-IDF

TF-IDF converts text into numerical features that can be understood by the Machine Learning model.

### Step 3: Logistic Regression

Logistic Regression is trained using the TF-IDF features.

### Step 4: Text Classification

The user can enter a sentence and the model predicts whether it belongs to:

- Tech
- Sports

The model also provides a confidence percentage.

### Step 5: File Analysis

The system can analyze a text file and classify each sentence.

### Step 6: File Statistics

The system calculates:

- Total lines
- Total words
- Total characters

### Step 7: Word Frequency

The system finds the most frequently used words in the file.

### Step 8: Category Summary

The system counts the number of Tech and Sports sentences and identifies the dominant category.

## Machine Learning Model

TF-IDF Vectorizer:
Converts text into numerical feature vectors.

Logistic Regression:
Classifies the text into Tech or Sports categories.

## Testing

The model is tested using `accuracy_test.py`.

The testing program displays:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## How to Run

First install the required library:

```bash
pip install scikit-learn