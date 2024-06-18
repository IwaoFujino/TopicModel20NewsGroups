# Apply topic model (LDA method) to word set of 20newsgroups and save results in pickle file
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import datetime

# main function
def main():
    # Load the word set of 20newsgroups dataset
    print("Loading the word set of 20newsgroups dataset...")
    with open("20newsgroups-docsdata.pickle", "rb") as f:
        docsdata = pickle.load(f)
    # In the document, count the number of occurrences for each word.
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    docscount = count_vectorizer.fit_transform(docsdata)
    print("Size of array docscount =", docscount.shape)
    vocabulary = count_vectorizer.get_feature_names_out()
    print("Total number of words in vocabulary =", len(vocabulary))
    vocabfile="20newsgroups-vocabulary.pickle"
    with open(vocabfile, "wb") as f1:
        pickle.dump(vocabulary, f1)
    print("Saved vocabulary into "+vocabfile+".")
    # Apply topic model to word set of 20newsgroups 
    nn_topics = 20
    print("Applying topic model (LDA method) to word set of 20newsgroups  ...")
    lda = LatentDirichletAllocation(n_components=nn_topics)
    lda.fit(docscount)
    print("Create a topic word matrix ...")
    topics = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    print("Size of the topic word matrix =", topics.shape)
    # Save topic extraction results
    topicfile="20newsgroups-topics.pickle"
    with open(topicfile, "wb") as f2:
        pickle.dump(topics, f2)
    print("Saved the topic word matrix into"+topicfile+".")
    return

# Run from here
if __name__ == "__main__":
	start_time = datetime.datetime.now()
	main()
	end_time = datetime.datetime.now()
	elapsed_time=end_time-start_time
	print("Elapsed time: ", elapsed_time)