# Morphological analysis of English documents: Split words and display tags
import nltk

with open("alice.txt","r", encoding="utf-8") as f:
    text = f.read()
print("Document:")
print(text)
words = nltk.word_tokenize(text)
print("Word:")
print(words)
wordtags = nltk.pos_tag(words)
print("Words and parts of speech:")
for word, tag in (wordtags):
    print("word=", word, "\t part of speech=", tag)