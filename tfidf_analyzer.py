from sklearn.feature_extraction.text import TfidfVectorizer
with open("sample.txt","r") as file:
    text=file.read()
vectorizer=TfidfVectorizer()
tfidf_matrix=vectorizer.fit_transform([text])
words=vectorizer.get_feature_names_out()
scores=tfidf_matrix.toarray()[0]
word_scores=list(zip(words,scores))
sorted_words=sorted(word_scores,key=lambda x:x[1],reverse=True)
print("="*50)
print(" TOP 5 IMPORTANT WORDS")
print("="*50)

for word,score in sorted_words[:5]:
    print(word,":",round(score,4))
print("\n"+"="*50)