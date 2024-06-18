# Extract nouns from 20 news groups, combine them into strings for each document, and save them in a list.
# Furthermore, write the list to a pickle file.
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
import pickle
import datetime

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Main function
def main():
    # Read data set
    print("Reading 20 news group dataset ...")
    dataset = fetch_20newsgroups(subset="all", random_state=0, remove=("headers", "footers", "quotes"))
    print("Attrbutes =", dir(dataset))
    for n in range(3):
        print("Document number=", n, "------------")
        print("Category =", dataset.target[n])
        print("Text =", dataset.data[n])
    print("Total number of documents in the dataset =", len(dataset.data))
    print("List of category names =", dataset.target_names)
    print("The number of category =", len(dataset.target_names))
    # Extract nouns from each document
    print("extracting nouns from each document ...")
    docsdata = []
    for docno, text in enumerate(dataset.data):
        words = []
        wordtags = nltk.pos_tag(nltk.word_tokenize(text))
        for word, tag in wordtags:
            if tag=='NN'or tag=='NNP' or tag=='NNS'or tag=='NNPS':
                lemma = lemmatizer.lemmatize(word)
                if(len(lemma)>=3): # more than 3 characters
                    words.append(lemma)
        if (len(words)>=5): # more than 5 words
            docdata = " ".join(words)
            docsdata.append(docdata)
    # Print the noun extraction results
    print("Printing the noun extraction results ...")
    for docno, docdata in enumerate(docsdata):
        print("Document number =", docno, "-----------------------")
        print("List of nouns =", docdata)
    # Save the noun extraction results
    docsfile = "20newsgroups-docsdata.pickle"
    with open(docsfile, "wb") as f:
        pickle.dump(docsdata, f)
    print("The word set was saved into "+docsfile+".")

# Run from here
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    elapsed_time = end_time-start_time
    print("Elapsed time: ", elapsed_time)