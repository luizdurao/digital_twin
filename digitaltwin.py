import pandas as pd
import nltk
from datetime import datetime
import datetime as dt
import emoji
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: 
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    
    summarize_text = []

    
    sentences =  read_article(file_name)

    
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

   
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    
    print("Summarize Text: \n", ". ".join(summarize_text))
    
    return summarize_text


def criacao(x):
    f = open('file.txt','w')
    f.write(str(x))
    f.close()
    try:
        summary = generate_summary( "file.txt", 2)
    except:
        return x
    return summary

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')

df=pd.read_csv("dt2.csv", sep=";")

df["resumo"] = df["AB"].apply(lambda x: criacao(x))

df["resumo2"]=df["resumo"].apply(lambda x: str(x))

df.loc[((df["DE"].str.contains("digital twin")) | (df["DE"].str.contains("digital factory")) | (df["DE"].str.contains("digital thread"))| (df["DE"].str.contains("digital shadow"))| (df["DE"].str.contains("smart factory"))| (df["DE"].str.contains("smart manufacturing"))),"manter"]=1

df2 = df.loc[df["manter"]==1]

tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = 'english'
)
tfidf.fit(df2.resumo2)
text = tfidf.transform(df2.resumo2)

find_optimal_clusters(text, 10)

clusters = MiniBatchKMeans(n_clusters=5, init_size=1024, batch_size=2048, random_state=20,compute_labels=True).fit_predict(text)

df2["cluster3"]=" "

df2["cluster3"] = clusters

df2.to_csv("manter2.csv", sep=";", index=False)

max_label = max(clusters)
max_items = np.random.choice(range(text.shape[0]), size=100, replace=False)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(text[max_items,:].todense())
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

df_cluster = pd.DataFrame(clusters, columns=['cluster'])
finalDf = pd.concat([principalDf, df_cluster["cluster"]], axis = 1)

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2,4]
colors = ['r', 'g', 'b','y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['cluster'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(["Cluster1","Cluster2","Cluster3","Cluster4"])
ax.grid()

for i in range(1,5):
    text= open("cluster"+ i + ".txt", encoding="utf8").read()
    wordcloud = WordCloud(max_font_size=100,width = 1520, height = 535).generate(text)
    plt.figure(figsize=(16,9))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

#https://gist.github.com/edubey/cc41dbdec508a051675daf8e8bba62c5