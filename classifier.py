from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter


# ----------------------------------------
# 1. LOAD TRAINING DATA
# ----------------------------------------

def load_training_data():
    texts = []
    labels = []

    # Load technology training data
    with open("data/tech.txt", "r") as file:
        for line in file:
            line = line.strip()

            if line:
                texts.append(line)
                labels.append("tech")

    # Load sports training data
    with open("data/sports.txt", "r") as file:
        for line in file:
            line = line.strip()

            if line:
                texts.append(line)
                labels.append("sports")

    return texts, labels


# ----------------------------------------
# 2. LOAD TRAINING DATA
# ----------------------------------------

texts, labels = load_training_data()


# ----------------------------------------
# 3. CONVERT TEXT INTO TF-IDF FEATURES
# ----------------------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)


# ----------------------------------------
# 4. CREATE AND TRAIN MODEL
# ----------------------------------------

model = LogisticRegression()

model.fit(X, labels)


# ----------------------------------------
# 5. CLASSIFY A SENTENCE
# ----------------------------------------

def classify_text(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    probability = model.predict_proba(text_vector)

    confidence = probability.max()

    return prediction[0], confidence


# ----------------------------------------
# 6. ANALYZE A FILE
# ----------------------------------------

def analyze_file(filename):

    with open(filename, "r") as file:
        lines = file.readlines()

    print("\n" + "=" * 50)
    print("             FILE ANALYZER")
    print("=" * 50)

    for line in lines:

        line = line.strip()

        if line:
            category, confidence = classify_text(line)

            print("\nSentence:", line)
            print("Predicted category:", category)
            print(
                "Confidence:",
                round(confidence * 100, 2),
                "%"
            )

    print("=" * 50)


# ----------------------------------------
# 7. FILE STATISTICS
# ----------------------------------------

def file_statistics(filename):

    with open(filename, "r") as file:
        text = file.read()

    lines = text.splitlines()
    words = text.split()
    characters = len(text)

    print("\n" + "=" * 50)
    print("             FILE STATISTICS")
    print("=" * 50)

    print("Total lines:", len(lines))
    print("Total words:", len(words))
    print("Total characters:", characters)

    print("=" * 50)


# ----------------------------------------
# 8. WORD FREQUENCY
# ----------------------------------------

def word_frequency(filename):

    with open(filename, "r") as file:
        text = file.read()

    words = text.split()

    word_count = Counter(words)

    print("\n" + "=" * 50)
    print("             WORD FREQUENCY")
    print("=" * 50)

    for word, count in word_count.most_common(10):
        print(f"{word}: {count}")

    print("=" * 50)


# ----------------------------------------
# 9. CATEGORY SUMMARY
# ----------------------------------------

def category_summary(filename):

    tech_count = 0
    sports_count = 0

    with open(filename, "r") as file:
        lines = file.readlines()

    for line in lines:

        line = line.strip()

        if line:

            category, confidence = classify_text(line)

            if category == "tech":
                tech_count += 1

            elif category == "sports":
                sports_count += 1

    print("\n" + "=" * 50)
    print("             CATEGORY SUMMARY")
    print("=" * 50)

    print("Tech articles:", tech_count)
    print("Sports articles:", sports_count)

    if tech_count > sports_count:
        print("Dominant category: Tech")

    elif sports_count > tech_count:
        print("Dominant category: Sports")

    else:
        print("Dominant category: Equal")

    print("=" * 50)


# ----------------------------------------
# 10. MAIN MENU
# ----------------------------------------

if __name__ == "__main__":

    while True:

        print("\n" + "=" * 50)
        print("              TEXT ANALYZER")
        print("=" * 50)

        print("1. Analyze a sentence")
        print("2. Analyze a file")
        print("3. File statistics")
        print("4. Word frequency")
        print("5. Category summary")
        print("6. Exit")

        choice = input("\nEnter your choice: ")


        # Option 1
        if choice == "1":

            new_text = input("\nEnter a sentence: ")

            category, confidence = classify_text(new_text)

            print("\nPredicted category:", category)
            print(
                "Confidence:",
                round(confidence * 100, 2),
                "%"
            )


        # Option 2
        elif choice == "2":

            analyze_file("sample.txt")


        # Option 3
        elif choice == "3":

            file_statistics("sample.txt")


        # Option 4
        elif choice == "4":

            word_frequency("sample.txt")


        # Option 5
        elif choice == "5":

            category_summary("sample.txt")


        # Option 6
        elif choice == "6":

            print("\nExiting Text Analyzer...")
            break


        # Invalid option
        else:

            print("\nInvalid choice! Please enter 1 to 6.")