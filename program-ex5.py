# Display the top words of each topic in a bar graph
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load saved vocabulary lists and topic word matrices
print("Loading vocabulary ...")
with open("20newsgroups-vocabulary.pickle", "rb") as f1:
    vocabulary = pickle.load(f1)
print("Loading the topic word matrix...")
with open("20newsgroups-topics.pickle", "rb") as f2:
    topics = pickle.load(f2)
# Display the top words for each topic
nn = 10
for topicno, topic in enumerate(topics):
    print("Topic number =", topicno)
    idx = np.argsort(-topic)[:nn]
    for id in idx:
        print(vocabulary[id], "\t", np.round(topic[id], 6))
# Create a bar graph of top words for each topic
kk, mm = topics.shape
figrow = 5 # Number of rows per page
figcol = 2 # Number of columns per page
for figno in range(kk//(figrow*figcol)):
    print("Page number of figures =", figno)
    fig, axes = plt.subplots(figrow, figcol, figsize=(28, 35), sharex=True)
    for topic_idx, topic in enumerate(topics[figno*figrow*figcol:(figno+1)*figrow*figcol]):
        idx = np.argsort(-topic)[:nn]
        top_words = [vocabulary[i] for i in idx]
        values = topic[idx]
        m = topic_idx // figcol
        n = topic_idx % figcol
        axes[m, n].barh(top_words, values, height=0.7)
        axes[m, n].set_title(f"Topic {topic_idx + figno*figrow*figcol}", fontdict={"fontsize": 30})
        axes[m, n].invert_yaxis()
        axes[m, n].set_xlim(0.00, 0.20)
        axes[m, n].tick_params(labelsize=24)
    fig.suptitle("Top words of each topic", fontsize=50)
    plt.subplots_adjust(top=0.94, bottom=0.05, wspace=0.90, hspace=0.3)
    topsfile = "20newsgroups-topwords-fig"+str(figno)+".png"
    plt.savefig(topsfile)
    print("The processing result was saved in "+topsfile+".")
