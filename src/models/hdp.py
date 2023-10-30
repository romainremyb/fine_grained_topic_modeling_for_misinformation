from .abstract_model import AbstractModel
import numpy as np
import os, nltk
from gensim.models import LdaModel, LdaMulticore, HdpModel
import logging
from collections import defaultdict
from utils import preprocess_for_bow, preprocess

ROOT = '.'

def get_indexes(topics_pred, threshold):
    return [(i, topics_pred[i]) for i in np.where(topics_pred>threshold)[0]]

class HDPwrapper(AbstractModel):
    def __init__(self, bow_corpus, id2word): #TODO check the overchunking_factor -> setting as 256 should give simiular results as with None -> check with a range of factors from (1 to 256)
        super().__init__(bow_corpus, id2word, None)
        #https://radimrehurek.com/gensim/models/hdpmodel.html
        self.model = HdpModel(bow_corpus, id2word,
                max_chunks=None, #Upper bound on how many chunks to process (retakes chunks from beginning of corpus if not enough docs)
                                        # if None -> same as len(corpus)/chunksize
                chunksize=256, #docs per chunks
                max_time=None, #upper bound on time for training model
                kappa=1.0, #exponentital decay factor for learning on batches
                tau=64.0, #downweight early iterations of documents
                K=15, # Second level truncation level
                T=150, #Top level truncation level
                alpha=1, #Second level concentration
                gamma=1, #first level concentration
                eta=0.01, #topic Dirichlet prior 
                scale=1.0, #Weights information from the mini-chunk of corpus to calculate rhot
                var_converge=0.0001, # Lower bound on the right side of convergence. Used when updating variational parameters for a single document
                outputdir=None, 
                random_state=None)
        
        self.num_topics=self.max_topics()
        

    def predict_rawtext(self, text, minimum_probability=0, preprocessing=True, 
                        preproc_params = {'keep_unicodes': {'keep': True, 'min_count_in_corpus': 2}, 'strip_brackets': False, 
                                          'add_adj_nn_pairs': True, 'verbs': True, 'adjectives': False}):
        if preprocessing:
            data=preprocess(text,  preproc_params['strip_brackets'], preproc_params['keep_unicodes']['keep'], 
                           preproc_params['add_adj_nn_pairs'],  preproc_params['verbs'], 
                           preproc_params['adjectives']) 
            tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            tokenized_data = [token.strip() for token in tokenizer.tokenize(data)]
        else:
            tokenized_data=text.split(' ')
        return self.get_document_topics(self.id2word.doc2bow(tokenized_data), minimum_probability=0) 
                            
    

    def predict_corpus(self, datapath, minimum_probability=None,   
                        preproc_params = {'keep_unicodes': {'keep': True, 'min_count_in_corpus': 2}, 'strip_brackets': False, 
                                          'add_adj_nn_pairs': True, 'verbs': True, 'adjectives': False}):
            """Predict over a corpus, using already trained model with its associated id2word dictionary
            """

            tokenized_data = preprocess_for_bow(datapath, return_idxs=False, preprocessing=True, preproc_params=preproc_params)['tokenized_data']
            bow = [self.id2word.doc2bow(seq) for seq in tokenized_data]
            return self.get_document_topics(self, bow, minimum_probability=minimum_probability)

    def get_document_topics(self, bow, minimum_probability=0.5):
        """
        bow can be List of doc bows or just one document bow
        """
        if type(bow[0])==list:
            return [get_indexes(np.array(self.model.inference([doc]))[0], minimum_probability) for doc in bow]
        else:
            return [get_indexes(np.array(self.model.inference([bow]))[0], minimum_probability)]
    
    def get_indexes_per_topics(self, bow_corpus, minimum_probability, index_list):
        result=defaultdict(list)
        generator = self.get_document_topics(bow_corpus, minimum_probability=minimum_probability)
        if len(bow_corpus)!=len(generator):
            raise NameError('Something wrong with document topic generator')
        for i in range(len(generator)):
            for topic in generator[i]:
                result[str(topic[0])].append(index_list[i])
        return result


    def max_topics(self):
        return self.model.get_topics().shape[0]

    def topics(self, topn=10):
        return [self.topic(i, topn) for i in range(0, self.max_topics())]

    def topic(self, topic_id: int, topn=10):
        if self.model is None:
            self.load()
        words = []
        weights = []
        for word, weight in self.model.show_topic(topic_id, topn=topn):
            weights.append(weight)
            words.append(word)
        return {
            'words': words,
            'weights': weights
        }