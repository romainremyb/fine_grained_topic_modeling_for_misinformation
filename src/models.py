import numpy as np
import os
from gensim.models import LdaModel, LdaMulticore, HdpModel
import logging
from collections import defaultdict

ROOT = '.'

class AbstractModel:
    def __init__(self, bow_corpus, id2word, index_list=None):
        self.bow_corpus=bow_corpus
        self.id2word=id2word
        self.index_list=index_list


    def get_document_topics(self, bow, minimum_probability=None):
        """
        returns List[Tuple[relevantTopicID, probability]]
        """
        raise NotImplementedError
    
    def get_indexes_per_topics(self, bow_corpus, minimum_probability, index_list):
        """
        returns Dict[str(topic_id)]=List[IDs in index_list] -> IDs are appended to each topic given minimum probability
        """
        raise NotImplementedError
    
    def get_term_topics(self, word_id, minimum_probability=1.e-20):
        """
        returns List[Tuple[relevantTopicID, probability]]
        """
        raise NotImplementedError
    
    

"""
metrics

"""



class LDAwrappers(AbstractModel):
    def __init__(self, bow_corpus, id2word, model, num_topics, decay=0.5, chunksize=2000, gamma_threshold=0.001):
        super().__init__(bow_corpus, id2word, None)
        modelnames=['LdaModelGensim', 'LdaMulticoreGensim']
        if model=="LdaModelGensim":
            self.model = LdaModel(bow_corpus, num_topics=num_topics,
                        id2word= id2word,
                        distributed=False,
                        chunksize=chunksize, #training chunks
                        decay=decay, # rate at which previous lambda value is forgotten (0.5,1)
                        passes=1, #training epochs
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
                        passes=1, #training epochs
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
     

    def get_document_topics(self, bow, minimum_probability=None, minimum_phi_value=None,
                            per_word_topics=False):
        """
        bow can be List of doc bows or just one document bow
        """
        return self.model.get_document_topics(bow, minimum_probability, minimum_phi_value,
                            per_word_topics)
    
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
    


    @property
    def topics(self):
        if self.model is None:
            self.load()

        return [self.topic(i) for i in range(0, self.model.num_topics)]

    def topic(self, topic_id: int, topn: int):
        if self.model is None:
            self.load()
        words = []
        weights = []
        for word, weight in self.model.show_topic(topic_id, topn=10):
            weights.append(weight)
            words.append(word)
        return {
            'words': words,
            'weights': weights
        }



class HDPwrapper(AbstractModel):
    def __init__(self, bow_corpus, id2word):
        super().__init__(bow_corpus, id2word, None)
        #https://radimrehurek.com/gensim/models/hdpmodel.html
        self.model = HdpModel(bow_corpus, id2word,
                            max_chunks=None, #Upper bound on how many chunks to process (retakes chunks from beginning of corpus if not enough docs)
                                                #! if (max_chunks, max_time)==None, max_chunks=m_D=len(corpus) -> means chunksize passes over the corpus !?
                                                # set as factor of len(corpus)?
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

