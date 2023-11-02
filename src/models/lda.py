from .abstract_model import AbstractModel
import numpy as np
import os, nltk
from gensim.models import LdaModel, LdaMulticore
from collections import defaultdict
from utils import preprocess_for_bow, preprocess

ROOT = '.'

class LDAwrappers(AbstractModel):
    def __init__(self, bow_corpus, id2word, model, num_topics, decay=0.5, passes=1, chunksize=2000, gamma_threshold=0.001):
        super().__init__(bow_corpus, id2word, None)
        modelnames=['LdaModelGensim', 'LdaMulticoreGensim']
        self.num_topics=num_topics
        
        if model=="LdaModelGensim":
            self.model = LdaModel(bow_corpus, num_topics=num_topics,
                        id2word= id2word,
                        distributed=False,
                        chunksize=chunksize, #training chunks
                        decay=decay, # rate at which previous lambda value is forgotten (0.5,1)
                        passes=passes, #training epochs
                        update_every=0, #number of documents to be iterated through for each update (during model deployement: set 0 if only need batch training over given corpus)
                        alpha='auto', #document/topic priors - array | symmetric=1/num_topics | 'asymmetric'=(topic_index + sqrt(num_topics)) | 'auto': learns asymmetric from corpus (need distributed set to True) 
                        eta='auto', #topic-word  priors - shape (num_topics, num_words) or vector for equal priors accross words
                                        #asymmetric and auto possible but equal distrib across words
                        offset=1, #slow down first iter -> math:`\\tau_0` from `'Online Learning for LDA' -> 0 -> no slowing down
                        eval_every=10, #log perplexity -> needed for auto ? 
                        iterations=50, #maximum iter over corpus for inference 
                        gamma_threshold=gamma_threshold, #minimum change in the value of the gamma parameters to continue iterating
                        minimum_probability=0.01, #filter out topic prob lower than that
                        random_state=None,
                        ns_conf=None, #optional: for distributed learning
                        minimum_phi_value=0.01, #lowerbound for topic/word
                        per_word_topics=False, #if true: also return topic/words distrib when calling .get_document_topics(
                        callbacks=None,
                        dtype=np.float32)
            
        elif model=="LdaMulticoreGensim":
            self.model = LdaMulticore(corpus=bow_corpus, num_topics=num_topics, 
                        id2word= id2word,
                        workers=None, #all available if None
                        batch=True, #True for batch learning, False for online learning (streaming)
                        chunksize=2000, #training chunks
                        decay=0.5, # rate at which previous lambda value is forgotten (0.5,1)
                        passes=passes, #training epochs
                        alpha='auto', #document/topic priors - array | symmetric=1/num_topics | 'asymmetric'=(topic_index + sqrt(num_topics)) | 'auto': learns asymmetric from corpus (need distributed set to True) 
                        eta='auto', #topic-word  priors - shape (num_topics, num_words) or vector for equal priors accross words
                                        #asymmetric and auto possible but equal distrib across words
                        offset=1, #slow down first iter -> math:`\\tau_0` from `'Online Learning for LDA'
                        eval_every=10, #log perplexity -> needed for auto ? 
                        iterations=50, #maximum iter over corpus for inference 
                        gamma_threshold=0.001, #minimum change in the value of the gamma parameters to continue iterating
                        minimum_probability=0.01, #filter out topic prob lower than that
                        random_state=None,
                        ns_conf=None, #optional: for distributed learning
                        minimum_phi_value=0.01, #lowerbound for topic/word
                        per_word_topics=False, #if true: also return topic/words distrib when calling .get_document_topics(
                        callbacks=None,
                        dtype=np.float32)

        else:
            raise ValueError(f'Wrong model name! must be one of {modelnames}')
        
    
    def predict_rawtext(self, text, minimum_probability=None, minimum_phi_value=None, per_word_topics=False, preprocessing=True, 
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
        return self.get_document_topics(self.id2word.doc2bow(tokenized_data), minimum_probability=minimum_probability, 
                            minimum_phi_value=minimum_phi_value, per_word_topics=per_word_topics)
    
    def predict_corpus(self, datapath, minimum_probability=None, minimum_phi_value=None, per_word_topics=False,  
                        preproc_params = {'keep_unicodes': {'keep': True, 'min_count_in_corpus': 2}, 'strip_brackets': False, 
                                          'add_adj_nn_pairs': True, 'verbs': True, 'adjectives': False}):
            """Predict over a corpus, using already trained model with its associated id2word dictionary
            """

            tokenized_data = preprocess_for_bow(datapath, return_idxs=False, preprocessing=True, preproc_params=preproc_params)['tokenized_data']
            bow = [self.id2word.doc2bow(seq) for seq in tokenized_data]
            return self.get_document_topics(bow, minimum_probability=minimum_probability, 
                            minimum_phi_value=minimum_phi_value, per_word_topics=per_word_topics)

    def get_document_topics(self, bow, minimum_probability=None, minimum_phi_value=None,
                            per_word_topics=False):
        """
        bow can be List of doc bows or just one document bow
        """
        return list(self.model.get_document_topics(bow, minimum_probability, minimum_phi_value,
                            per_word_topics))
    
    def get_indexes_per_topics(self, bow_corpus, minimum_probability, index_list):
        result=defaultdict(list)
        generator = self.get_document_topics(bow_corpus, minimum_probability=minimum_probability)
        if len(bow_corpus)!=len(generator):
            raise NameError('Something wrong with document topic generator')
        for i in range(len(generator)):
            for topic in generator[i]:
                result[str(topic[0])].append(index_list[i])
        return result
    

    def get_term_topics(self, word_id, minimum_probability=1.e-20):
        return self.model.get_term_topics(word_id, minimum_probability)
    

    def topics(self, topn=10):
        return [self.topic(i, topn) for i in range(0, self.num_topics)]

    def topic(self, topic_id: int, topn=10):
        if self.model is None:
            self.load()
        words = []
        weights = []
        for word, weight in self.model.show_topic(topic_id, topn=topn):
            weights.append(float(weight))
            words.append(word)
        return {
            'words': words,
            'weights': weights
        }