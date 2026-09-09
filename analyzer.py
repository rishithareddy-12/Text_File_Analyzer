import string
def read_file(filename):
    file=open(filename,"r")
    text = file.read()
    file.close()
    return text
def clean_text(text):  
   
    text=text.lower()

    text=text.translate(str.maketrans("","",string.punctuation))
    return text
def count_words(words):



    word_count={}

    for word in words:
        if word in word_count:
            word_count[word]+=1
        else:
            word_count[word]=1
    return word_count
text=read_file("sample.txt")
lines = text.splitlines()
number_of_lines = len(lines)
number_of_characters = len(text)
cleaned_text=clean_text(text)
words=cleaned_text.split()
word_count=count_words(words)

sorted_words=sorted(word_count.items(),key=lambda x:x[1],reverse=True)
print("="*50)
print("   The text analysis of the file")
print("="*50)
print("Number of lines:",number_of_lines)
print("Number of characters:",number_of_characters)
print("Number of words:",len(words))

print("\nTop 5 most frequent words:")
for word,count in sorted_words[:5]:
        print(word,":",count)
print("word frequency:",word_count)
print("\n"+"="*50)
