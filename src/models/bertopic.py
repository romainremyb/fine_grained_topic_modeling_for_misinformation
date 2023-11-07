from .abstract_model import AbstractModel
import numpy as np
import os, nltk
from umap import UMAP
from gensim.models import LdaModel, LdaMulticore
from collections import defaultdict
from utils import preprocess_for_bow, preprocess
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

class BERTopicWrapper(AbstractModel):
    """https://github.com/MaartenGr/BERTopic/tree/master
    """
    def __init__(self, docs, IDs, nb_topics=None, top_words=10, min_topic_size=6, umap_param={'n_neighbors': 15, 'n_components': 5},
                    representation_model = None, custom_vectorizer=False, vectorizer_params = {
                        'strip_accents': None, 'stop_words': "english",
                        'ngram_range': (1,2)}): 
                    #TODO set min_df as fraction of nb of docs
        super().__init__(None, IDs, None) 
        """docs : Union[str,List]
        vectorizer: for computing c_tf_idf after doc clustering -> can be done before or after with update_topics

        """
        #TODO always pretrain with standard params ? vectorizer_params for update_topics ?
        self.vectorizer=CountVectorizer(strip_accents= vectorizer_params['strip_accents'],
                        stop_words= vectorizer_params['stop_words'],
                        ngram_range= vectorizer_params['ngram_range']) 
        
        if custom_vectorizer==True:
            self.model=BERTopic(language = "english",
                                top_n_words = top_words,
                                n_gram_range = (1, 1),
                                min_topic_size = min_topic_size,
                                nr_topics = nb_topics,
                                low_memory = False,
                                calculate_probabilities = False,
                                embedding_model = None, #if None and no embedding provided in fit()/fit_transform(), will use sentence_transformers.SentenceTransformer
                                vectorizer_model = self.vectorizer, # 
                                umap_model = UMAP(n_neighbors=umap_param['n_neighbors'],
                                                n_components=umap_param['n_components'],
                                                min_dist=0.0,
                                                metric='cosine',
                                                low_memory=False),
                                hdbscan_model = None, #sets min_topic_size
                                representation_model = representation_model)
        else:
            self.model=BERTopic(language = "english",
                                top_n_words = top_words,
                                n_gram_range = (1, 1),
                                min_topic_size = min_topic_size,
                                nr_topics = nb_topics,
                                low_memory = False,
                                calculate_probabilities = False,
                                embedding_model = None, #if None and no embedding provided in fit()/fit_transform(), will use sentence_transformers.SentenceTransformer
                                vectorizer_model = None, 
                                umap_model = UMAP(n_neighbors=umap_param['n_neighbors'],
                                                n_components=umap_param['n_components'],
                                                min_dist=0.0,
                                                metric='cosine',
                                                low_memory=False),
                                hdbscan_model = None, #sets min_topic_size
                                representation_model = representation_model)

        if type(docs)==str:
            if os.path.exists(docs)==False:
                raise ValueError('wrong datapath')
            else:
                data = preprocess_for_bow('data.csv', preprocessing=False)
                self.text = data['text']
                self.ids = data['ids']

        





        