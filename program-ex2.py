# Morphological analysis of English documents: extracting nouns and converting them to lemmas
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
with open("alice.txt","r", encoding="utf-8") as f:
    text = f.read()
print("Document:")
print(text)
words = nltk.word_tokenize(text)
print("Words:")
print(words)
wordtags = nltk.pos_tag(words)
docdata = []
print("Words >> tags:")
for word, tag in wordtags:
    print(word, ">>", tag)
    if tag=='NN'or tag=='NNP' or tag=='NNS'or tag=='NNPS':
        docdata.append(lemmatizer.lemmatize(word, 'n'))
print("List of nouns:")
print(docdata)