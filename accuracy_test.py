from classifier import load_training_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load data
texts, labels = load_training_data()


# 2. Split data into training and testing
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# 3. Convert text into TF-IDF
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# 4. Train model
model = LogisticRegression()

model.fit(X_train, y_train)


# 5. Make predictions
predictions = model.predict(X_test)


# 6. Calculate accuracy
accuracy = accuracy_score(y_test, predictions)


print("\n" + "=" * 50)
print("             MODEL ACCURACY")
print("=" * 50)

print("Accuracy:", round(accuracy * 100, 2), "%")


# 7. Classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions))


# 8. Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("=" * 50)