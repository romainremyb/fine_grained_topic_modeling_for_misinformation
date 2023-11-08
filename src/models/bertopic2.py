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
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired

"""
        keybert = KeyBERTInspired()
        mmr = MaximalMarginalRelevance(diversity=0.3)
        chained_representation = [keybert, mmr]
        -> change to default representation ?
"""


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


        super().__init__(None, top_n_words, (1, 1), min_topic_size, nb_topics, False, False, None, None, None, None, 
                         vectorizer, UMAP(n_neighbors=umap_param['n_neighbors'],
                                            n_components=umap_param['n_components'],
                                            min_dist=0.0,
                                            metric='cosine',
                                            low_memory=False), 
                        None, representation_model) 
        
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




    def set_topics(self, list_topic_str):
        self.topics_set_mapper = defaultdict(int)
        self.topics_set_embeddings = []
        for topic_name in list_topic_str:
            totalweight=0
            embedding=np.zeros(5)
            for i in range(len(self.find_topics(topic_name)[0])):
                idxs = [j+1 for j in range(len(self.topics_)) if self.topics_[j]==self.find_topics(topic_name)[0][i]] # assumes always label -1
                totalweight += self.find_topics(topic_name)[1][i]
                embedding = embedding + self.find_topics(topic_name)[1][i]*self.umap_embeddings[idxs].sum(axis=0)/len(idxs) 
            embedding = embedding/totalweight
            self.topics_set_mapper[topic_name] = i
            self.topics_set_embeddings.append({'embedding': embedding, 'topic_labels': self.find_topics(topic_name)[0],
                                             'topic_idxs': idxs}) #TODO: only save embedding?
            
    #def get_min_distance_set_topics(self): 
        #r=[]
        #for i in self.topics_set_embeddings:



