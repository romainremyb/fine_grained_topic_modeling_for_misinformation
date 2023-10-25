import numpy as np
import os
from gensim.models import LdaModel, LdaMulticore, HdpModel
import logging

ROOT = '.'

class AbstractModel:
    def __init__(self):
        pass

    def get_document_topics(self, bow, minimum_probability=None, minimum_phi_value=None,
                            per_word_topics=False):
        """
        returns List[Tuple[relevantTopicID, probability]]
        """
        raise NotImplementedError
    
    def get_term_topics(self, word_id, minimum_probability=1.e-20):
        """
        returns List[Tuple[relevantTopicID, probability]]
        """
        raise NotImplementedError
    
    def get_term_topic_distributions(self, num_words=10):
        """
        returns Dict[str(topic_label)] -> List[(term, probability), ...]
        """
        raise NotImplementedError




class LDAwrapper(AbstractModel):
    def __init__(self, model, corpus, dictionary, num_topics, decay=0.5, chunksize=2000, gamma_threshold=0.001):
        super().__init__()
        if model=="LdaModel":
            self.lda = LdaModel(corpus, num_topics=num_topics,
                        id2word= dictionary,
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
            
        elif model=="LdaMulticore":
            self.lda = LdaMulticore(corpus=corpus, num_topics=num_topics, 
                        id2word= dictionary,
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
            raise ValueError('Wrong model name!')
            return 

    def get_document_topics(self, bow, minimum_probability=None, minimum_phi_value=None,
                            per_word_topics=False):
        return self.lda.get_document_topics(bow, minimum_probability, minimum_phi_value,
                            per_word_topics)
    
    def get_term_topics(self, word_id, minimum_probability=1.e-20):
        return self.lda.get_term_topics(word_id, minimum_probability)
    

    def get_term_topic_distributions(self, num_words=10):
        result=dict()
        prints = self.lda.print_topics(num_words=num_words)
        for r in prints:
            result[str(r[0])] = []
            for term in r[1].split(' + '):
                result[str(r[0])].append((term.split('*')[1].replace('"',''), float(term.split('*')[0])))
        return result


    

