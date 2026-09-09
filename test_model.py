from classifier import classify_text


print("=" * 50)
print("        TEXT CLASSIFIER TEST")
print("=" * 50)

test_sentences = [
    "The new processor improves computer performance",
    "The football team won the championship",
    "Artificial intelligence is changing technology",
    "The player scored three goals in the match",
    "Python is used for machine learning",
    "The cricket team won the final match"
]

for sentence in test_sentences:

    category, confidence = classify_text(sentence)

    print("\nSentence:", sentence)
    print("Category:", category)
    print("Confidence:", round(confidence * 100, 2), "%")


print("\n" + "=" * 50)
print("Testing completed!")
print("=" * 50)