from .abstract_model import AbstractModel
import numpy as np
import pandas as pd
import os, nltk
from umap import UMAP
from gensim.models import LdaModel, LdaMulticore
from collections import defaultdict
from utils import preprocess_for_bow, preprocess
from bertopic import BERTopic
from bertopic.backend._utils import select_backend
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from scipy import stats




class BERTopicWrapper(BERTopic):
    """https://github.com/MaartenGr/BERTopic/tree/master
    """
    def __init__(self, docs, IDs, nb_topics=None, top_n_words=10, min_topic_size=6, umap_param={'n_neighbors': 15, 'n_components': 7},
                    representation_model = None, custom_vectorizer=False, vectorizer_params = {
                        'strip_accents': None, 'stop_words': "english",
                        'ngram_range': (1,2)}, embedding_model= None): 
        
        self.vectorizer=CountVectorizer(strip_accents= vectorizer_params['strip_accents'],
                        stop_words= vectorizer_params['stop_words'],
                        ngram_range= vectorizer_params['ngram_range']) 
        if custom_vectorizer==True:
            vectorizer=self.vectorizer
        else:
            vectorizer=None

        umap=UMAP(n_neighbors=umap_param['n_neighbors'],
                                                n_components=umap_param['n_components'],
                                                min_dist=0.0,
                                                metric='cosine',
                                                low_memory=False)

        super().__init__(None, top_n_words, (1, 1), min_topic_size, nb_topics, False, False, None, None, 
                         umap, None, vectorizer, None, representation_model) 
        
        """docs : Union[str,List]
        vectorizer: for computing c_tf_idf after doc clustering -> can be done before or after with update_topics

        """

        if type(docs)==str:
            if os.path.exists(docs)==False:
                raise ValueError('wrong datapath')
            else:
                data = preprocess_for_bow('data.csv', preprocessing=False)
                self.text = data['text'][1:]
                self.ids = data['ids'][1:]
        else:
            self.text = docs
            self.ids = IDs

        
        self.doc_ids = range(len(self.text)) 

        self.fit(self.text)

        documents = pd.DataFrame({"Document": self.text,
                                  "ID": self.doc_ids,
                                  "Topic": None,
                                  "Image": None})

        self.embedding_model = select_backend(embedding_model,
                                                  language="english")
        embeddings = self._extract_embeddings(documents.Document.values.tolist(),
                                        images=None,
                                        method="document",
                                        verbose=False)

        self.umap_embeddings = self._reduce_dimensionality(embeddings, None)

        #self.cosine_similarity = cosine_similarity(self.umap_embeddings)
        #self.normalized_cos_dist = 1 - self.cosine_similarity

        #self.dot_product_scores = np.dot(self.umap_embeddings, self.umap_embeddings.T)
        #self.normalized_dot_dist =  1 - (self.dot_product_scores / (np.linalg.norm(self.umap_embeddings, axis=1)[:, np.newaxis] * np.linalg.norm(self.umap_embeddings, axis=1)))


    def set_topics(self, list_topic_str, top_n, min_sim):
        """
        -> find possible docs belonging to topic labels -> aimed to generate new centroids
        returns self.topics_set_embeddings -> list of input topic labels containing:
                                - sbert_embedding: weighted average (on cosine similarity) of retrieved documents' embeddings - to be used as centroid
                                - ctfidf_embedding: weighted average (on cosine similarity) of retrieved documents' embeddings - to be used as centroid 
                                - topic_labels
                                - topic_idxs: doc idxs that were retrieved

        :param list_topic_str: list of input topic label
        :param top_n: slects the top_n clusters with smallest distance to input label c-tf-idf representation
        :param min_sim: further filter clusters by using a cosine similarity threshold
        """
        self.topics_set_mapper = defaultdict(int)
        self.topics_set_embeddings = []
        c = 0
        for topic_name in list_topic_str:
            totalweight=0
            sbert_embedding=np.zeros(self.umap_embeddings.shape[1])
            ctfidf_embedding=np.zeros(self.c_tf_idf_.shape[1])
            topics_found = self.find_topics(topic_name, top_n=top_n)
            topic_idxs=[]
            for i in range(len(topics_found[0])):
                if topics_found[1][i]<min_sim:
                    continue
                topic_idxs.append(topics_found[0][i])
                idxs = [j for j in range(len(self.topics_)) if self.topics_[j]==topics_found[0][i]] 
                totalweight += topics_found[1][i]
                sbert_embedding = sbert_embedding + topics_found[1][i]*self.umap_embeddings[idxs].sum(axis=0)/len(idxs) 
            ctfidf_embedding = ctfidf_embedding + topics_found[1][i]*self.c_tf_idf_[topics_found[0][i]].toarray().sum(axis=0)
            sbert_embedding = sbert_embedding/totalweight
            ctfidf_embedding = ctfidf_embedding/totalweight
            self.topics_set_mapper[topic_name] = c
            self.topics_set_embeddings.append({'sbert_embedding': sbert_embedding, 'ctfidf_embedding': ctfidf_embedding,
                                               'topic_labels': topic_idxs, 'topic_idxs': idxs}) 
            c+=1
            

    def sbert_cluster_set_topics(self):
        """ Uses centroids in self.topics_set_embeddings to recluster the corpus
        returns:
                - sbert_clusters: array containing topic labels (shape=len(corpus)), labels range(0,len(input labels))
                - sbert_centroids: sbert_centroids matrix shape(len(input_labels), len(sbert_embedding))
                - sbert_distances: norm distance of documents to their closest centroid ( shape=(len(corpus)) )
        """
        # create set topic embedding matrix
        centroids = np.array([self.topics_set_embeddings[0]['sbert_embedding']]) 
        for i in range(1, len(self.topics_set_embeddings)): 
            centroids=np.vstack((centroids,self.topics_set_embeddings[i]['sbert_embedding']))
        # loop into topic labels
        topics=[]
        for i in range(self.umap_embeddings.shape[0]):
            topic_norms = []
            for j in range(len(centroids)):        
                norm = np.linalg.norm(self.umap_embeddings[i,:]-centroids[j,:])
                topic_norms.append(norm)
            best = np.argmin(topic_norms)
            topics.append(best)
        self.sbert_clusters = np.array(topics)
        self.sbert_centroids=centroids
        self.sbert_distances = np.linalg.norm(self.umap_embeddings - self.sbert_centroids[self.sbert_clusters], axis=1)


    def ctfidf_cluster_set_topics(self):
        """ Uses ctfidf centroids in self.topics_set_embeddings to recluster the corpus
        returns:
                - ctfidf_clusters: array containing topic labels (shape=len(corpus)), labels range(0,len(input labels))
                - ctfidf_centroids: sbert_centroids matrix shape(len(input_labels), len(ctfidf_embedding))
                - ctfidf_distances: norm distance of documents to their closest centroid ( shape=(len(corpus)) )
        """
        centroids = np.array([self.topics_set_embeddings[0]['ctfidf_embedding']]) 
        for i in range(1, len(self.topics_set_embeddings)): # create set topic embedding matrix
            centroids=np.vstack((centroids,self.topics_set_embeddings[i]['ctfidf_embedding']))

        topics=[]
        for i in range(self.c_tf_idf_.shape[0]):
            topic_norms = []
            for j in range(len(centroids)):        
                norm = np.linalg.norm(self.c_tf_idf_[i,:].toarray()-centroids[j,:])
                topic_norms.append(norm)
            best = np.argmin(topic_norms)
            topics.append(best)
        self.ctfidf_clusters = np.array(topics)
        self.ctfidf_centroids=centroids
        self.ctfidf_distances = np.linalg.norm(self.c_tf_idf_.toarray() - self.ctfidf_centroids[self.ctfidf_clusters], axis=1)



    def get_sbert_z_outliers(self, z_threshold): 
        """
        returns: List of List of doc idxs where sbert-distance to input label centroid is above a z-threshold 
        """
        outliers_idx = []
        for i in range(self.sbert_centroids.shape[0]):
            idxs = np.where(np.array(self.sbert_clusters)==i)[0] # indexes of topic i
            z_scores = stats.zscore(self.sbert_distances[idxs])
            outliers_idx.append(idxs[np.where(np.abs(z_scores) > abs(z_threshold))[0].tolist()].tolist())
        return outliers_idx


    def get_sbert_norm_outliers(self, d_threshold): 
        """
        returns: List of List of doc idxs where ctfidf-distance to input label centroid is above a z-threshold 
        """
        outliers_idx = []
        for i in range(len(self.sbert_centroids)):
            idxs = np.where(np.array(self.sbert_clusters)==i)[0] # indexes of topic i
            outliers_idx.append(idxs[np.where(np.abs(self.sbert_distances[idxs]) > abs(d_threshold))[0].tolist()].tolist())
        return outliers_idx
    




