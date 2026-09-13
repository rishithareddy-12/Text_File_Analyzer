import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter


# ----------------------------------------
# PAGE SETTINGS
# ----------------------------------------

st.set_page_config(
    page_title="Text File Analyzer",
    page_icon="📄"
)

st.title("📄 Text File Analyzer")
st.write("Analyze and classify text using Machine Learning.")


# ----------------------------------------
# LOAD TRAINING DATA
# ----------------------------------------

def load_training_data():

    texts = []
    labels = []

    with open("data/tech.txt", "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                texts.append(line)
                labels.append("tech")

    with open("data/sports.txt", "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                texts.append(line)
                labels.append("sports")

    return texts, labels


texts, labels = load_training_data()


# ----------------------------------------
# TRAIN MODEL
# ----------------------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

model = LogisticRegression()

model.fit(X, labels)


# ----------------------------------------
# SENTENCE CLASSIFICATION
# ----------------------------------------

st.header("🔍 Analyze a Sentence")

sentence = st.text_input("Enter a sentence:")

if st.button("Analyze Sentence"):

    if sentence.strip():

        text_vector = vectorizer.transform([sentence])

        prediction = model.predict(text_vector)

        probability = model.predict_proba(text_vector)

        confidence = probability.max()

        st.success(f"Predicted Category: {prediction[0].title()}")

        st.info(
            f"Confidence: {confidence * 100:.2f}%"
        )

    else:

        st.warning("Please enter a sentence.")


# ----------------------------------------
# FILE ANALYSIS
# ----------------------------------------

st.header("📁 Analyze Sample File")

if st.button("Analyze File"):

    with open("sample.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:

        line = line.strip()

        if line:

            text_vector = vectorizer.transform([line])

            prediction = model.predict(text_vector)

            probability = model.predict_proba(text_vector)

            confidence = probability.max()

            st.write(f"**Sentence:** {line}")
            st.write(
                f"**Category:** {prediction[0].title()}"
            )
            st.write(
                f"**Confidence:** {confidence * 100:.2f}%"
            )
            st.divider()


# ----------------------------------------
# FILE STATISTICS
# ----------------------------------------

st.header("📊 File Statistics")

if st.button("Show Statistics"):

    with open("sample.txt", "r", encoding="utf-8") as file:
        text = file.read()

    lines = text.splitlines()
    words = text.split()
    characters = len(text)

    col1, col2, col3 = st.columns(3)

    col1.metric("Lines", len(lines))
    col2.metric("Words", len(words))
    col3.metric("Characters", characters)


# ----------------------------------------
# WORD FREQUENCY
# ----------------------------------------

st.header("🔤 Word Frequency")

if st.button("Show Word Frequency"):

    with open("sample.txt", "r", encoding="utf-8") as file:
        text = file.read()

    words = text.split()

    word_count = Counter(words)

    for word, count in word_count.most_common(10):

        st.write(f"**{word}** : {count}")


# ----------------------------------------
# CATEGORY SUMMARY
# ----------------------------------------

st.header("📈 Category Summary")

if st.button("Show Category Summary"):

    tech_count = 0
    sports_count = 0

    with open("sample.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:

        line = line.strip()

        if line:

            text_vector = vectorizer.transform([line])

            prediction = model.predict(text_vector)

            if prediction[0] == "tech":
                tech_count += 1

            elif prediction[0] == "sports":
                sports_count += 1

    st.write(f"💻 **Tech articles:** {tech_count}")
    st.write(f"⚽ **Sports articles:** {sports_count}")

    if tech_count > sports_count:
        st.success("Dominant Category: Tech")

    elif sports_count > tech_count:
        st.success("Dominant Category: Sports")

    else:
        st.info("Dominant Category: Equal")