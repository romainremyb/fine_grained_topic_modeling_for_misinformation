from .abstract_model import AbstractModel
import numpy as np
from numpy.random import multinomial
from numpy import log, exp
from numpy import argmax
import json, nltk
from collections import defaultdict
from utils import preprocess_for_bow, preprocess
import logging

def get_indexes(topics_pred, threshold):
    return [(i, topics_pred[i]) for i in np.where(topics_pred>threshold)[0]]


class MovieGroupProcessWrapper(AbstractModel):
    def __init__(self, bow_corpus, id2word, K=8, alpha=0.1, beta=0.1, n_iters=30):
        '''
        A MovieGroupProcess is a conceptual model introduced by Yin and Wang 2014 to
        describe their Gibbs sampling algorithm for a Dirichlet Mixture Model for the
        clustering short text documents.
        Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        Imagine a professor is leading a film class. At the start of the class, the students
        are randomly assigned to K tables. Before class begins, the students make lists of
        their favorite films. The teacher reads the role n_iters times. When
        a student is called, the student must select a new table satisfying either:
            1) The new table has more students than the current table.
        OR
            2) The new table has students with similar lists of favorite movies.

        :param K: int
            Upper bound on the number of possible clusters. Typically many fewer
        :param alpha: float between 0 and 1
            Alpha controls the probability that a student will join a table that is currently empty
            When alpha is 0, no one will join an empty table.
        :param beta: float between 0 and 1
            Beta controls the student's affinity for other students with similar interests. A low beta means
            that students desire to sit with students of similar interests. A high beta means they are less
            concerned with affinity and are more influenced by the popularity of a table
        :param n_iters:
        '''
        super().__init__(bow_corpus, id2word, None)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters

        # slots for computed variables
        self.number_docs = None
        self.vocab_size = None
        self.cluster_doc_count = [0 for _ in range(K)]
        self.cluster_word_count = [0 for _ in range(K)]
        self.cluster_word_distribution = [{} for i in range(K)]
        
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.debug('start training GSDMM')
        self.fit(bow_corpus, len(id2word))
        self.log.debug('end training GSDMM')
        self.num_topics=self.K


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
        return [get_indexes(np.array(self.score(doc_bow)), float(minimum_probability)) for doc_bow in bow]

    def get_indexes_per_topics(self, bow_corpus, minimum_probability, index_list):
        result=defaultdict(list)
        generator = self.get_document_topics(bow_corpus, minimum_probability=minimum_probability)
        if len(bow_corpus)!=len(generator):
            raise NameError('Something wrong with document topic generator')
        for i in range(len(generator)):
            for topic in generator[i]:
                result[str(topic[0])].append(index_list[i])
        return result
    

    def topics(self, topn=10):
        topics = []
        for i, topic in enumerate(self.cluster_word_distribution):
            current_words = []
            current_freq = []
            total = sum(topic.values())
            for word, freq in sorted(topic.items(), key=lambda item: item[1], reverse=True)[:topn]:
                current_words.append(self.id2word[word[0]])
                current_freq.append(freq / total)
            if len(current_words)>0:
                topics.append({
                    'words': current_words,
                    'weights': current_freq
                })
        return topics

    def topic(self, topic_id: int, topn=10):
        return self.topics(topn)[topic_id]


    @staticmethod
    def _sample(p):
        '''
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the multinomial distribution
        :return: int
            index of randomly selected output
        '''
        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]

    def fit(self, docs, vocab_size):
        '''
        Cluster the input documents
        :param docs: list of list
            list of lists containing the unique token set of each document
        :param V: total vocabulary size for each document
        :return: list of length len(doc)
            cluster label for each document
        '''
        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, vocab_size

        D = len(docs)
        self.number_docs = D
        self.vocab_size = vocab_size

        # unpack to easy var names
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution
        cluster_count = K
        d_z = [None for i in range(len(docs))]

        # initialize the clusters
        for i, doc in enumerate(docs):

            # choose a random  initial cluster for the doc
            z = self._sample([1.0 / K for _ in range(K)])
            d_z[i] = z
            m_z[z] += 1
            n_z[z] += len(doc)

            for word in doc:
                if word not in n_z_w[z]:
                    n_z_w[z][word] = 0
                n_z_w[z][word] += 1

        for _iter in range(n_iters):
            total_transfers = 0

            for i, doc in enumerate(docs):

                # remove the doc from it's current cluster
                z_old = d_z[i]

                m_z[z_old] -= 1
                n_z[z_old] -= len(doc)

                for word in doc:
                    n_z_w[z_old][word] -= 1

                    # compact dictionary to save space
                    if n_z_w[z_old][word] == 0:
                        del n_z_w[z_old][word]

                # draw sample from distribution to find new cluster
                p = self.score(doc)
                z_new = self._sample(p)

                # transfer doc to the new cluster
                if z_new != z_old:
                    total_transfers += 1

                d_z[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)

                for word in doc:
                    if word not in n_z_w[z_new]:
                        n_z_w[z_new][word] = 0
                    n_z_w[z_new][word] += 1

            cluster_count_new = sum([1 for v in m_z if v > 0])
            print("In stage %d: transferred %d clusters with %d clusters populated" % (
            _iter, total_transfers, cluster_count_new))
            if total_transfers == 0 and cluster_count_new == cluster_count and _iter>25:
                print("Converged.  Breaking out.")
                break
            cluster_count = cluster_count_new
        self.cluster_word_distribution = n_z_w
        return d_z

    def score(self, doc):
        '''
        Score a document

        Implements formula (3) of Yin and Wang 2014.
        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        :param doc: list[str]: The doc token stream
        :return: list[float]: A length K probability vector where each component represents
                              the probability of the document appearing in a particular cluster
        '''
        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution

        p = [0 for _ in range(K)]

        #  We break the formula into the following pieces
        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        #  lN1 = log(m_z[z] + alpha)
        #  lN2 = log(D - 1 + K*alpha)
        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))
        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))

        lD1 = log(D - 1 + K * alpha)
        doc_size = len(doc)
        for label in range(K):
            lN1 = log(m_z[label] + alpha)
            lN2 = 0
            lD2 = 0
            for word in doc:
                lN2 += log(n_z_w[label].get(word, 0) + beta)
            for j in range(1, doc_size +1):
                lD2 += log(n_z[label] + V * beta + j - 1)
            p[label] = exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm>0 else 1
        return [pp/pnorm for pp in p]

    def choose_best_label(self, doc):
        '''
        Choose the highest probability label for the input document
        :param doc: list[str]: The doc token stream
        :return:
        '''
        p = self.score(doc)
        return argmax(p),max(p)