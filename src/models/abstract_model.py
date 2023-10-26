import numpy as np
import os, pickle
import gensim
from gensim.models import KeyedVectors
from gensim.test import utils
from gensim.scripts.glove2word2vec import glove2word2vec
from collections import defaultdict
from utils import preprocess_for_bow, preprocess

ROOT = '.'

class AbstractModel:
    def __init__(self, bow_corpus, id2word, index_list=None):
        self.bow_corpus=bow_corpus
        self.id2word=id2word
        self.index_list=index_list

    def load(self, path=None):
        """
            Load the model and eventual dependencies.

            :param path: Folder where the model to be loaded is. If not specified, a default one is assigned
        """
        #  Implementation not mandatory.
        if path is not None:
            self.model_path = path

    def save(self, path=None):
        """
            Save the model and eventual dependencies.

            :param path: Folder where to save the model. If not specified, a default one is assigned
        """
        if path is not None:
            self.model_path = path

#TODO: topN or p-values
    # Perform Inference
    def predict_rawtext(self, text, topn=5, preprocessing=True, preproc_params=
                       {'keep_unicodes': {'keep': True, 'min_count_in_corpus': 2},
                        'strip_brackets': False, 'add_adj_nn_pairs': True, 'verbs': True, 
                        'adjectives': False}):
        """Predict topic for a given text

            :param text: The text on which performing the prediction
            :param int topn: Number of most probable topics to return
            :param bool preprocess: If True, execute preprocessing on the document
        """

        tokenized_data=preprocess(text) 
        bow = [self.id2word.doc2bow(seq) for seq in tokenized_data]
        
        raise NotImplementedError

#TODO: topN or p-values
    def predict_corpus(self, datapath, topn=5):
        """Predict over a corpus, using already trained model with its associated id2word dictionary
        """
        if self.model is None:
            self.load()

        tokenized_data = preprocess_for_bow(datapath)['tokenized_data']
        bow = [self.id2word.doc2bow(seq) for seq in tokenized_data]

        raise NotImplementedError
    

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
    
    
    @property
    def topics(self):
        """ List of the topics computed by the model

            :returns: a list of topic objects containing
            - 'words' the list of words related to the topic
            - 'weights' of those words in order (not always present)
        """
        raise NotImplementedError

    def topic(self, topic_id: int):
        """ Get info on a given topic

            :param int topic_id: Id of the topic
            :returns: an object containing
            - 'words' the list of words related to the topic
            - 'weights' of those words in order (not always present)
        """

        return self.topics[topic_id]
    

    def coherence(self, datapath, metrics=['c_v', 'c_npmi', 'c_uci', 'u_mass', 'c_we'], glove_path='glove/glove.6B.300d.txt'):
        """ Get the coherence of the topic mode.

        :param datapath: Path of the corpus on which compute the coherence.
        :param metric: Metric for computing the coherence, among <c_v, c_npmi, c_uci, u_mass, c_we>
         """
        results = defaultdict(dict)
        for metric in metrics:
            if metric not in ['c_v', 'c_npmi', 'c_uci', 'u_mass', 'c_we']:
                raise RuntimeError('Unrecognised metric: ' + metric)

            topic_words = [x['words'] for x in self.topics]

            self.log.debug('loading dataset')
            with open(datapath, "r") as datafile:
                text = [line.rstrip().split() for line in datafile if line]

            dictionary = gensim.corpora.hashdictionary.HashDictionary(text)


            if metric == 'c_we':

                if os.path.exists(glove_path.replace('txt', 'pickle')):
                    glove = pickle.load(open(glove_path.replace('txt', 'pickle'), 'rb'))
                else:
                    w2v = utils.get_tmpfile("w2v")
                    glove2word2vec(glove_path, w2v)
                    glove = KeyedVectors.load_word2vec_format(w2v)
                    pickle.dump(glove, open(glove_path.replace('txt', 'pickle'), 'wb'))

                results[metric]['c_we_per_topic'] = []

                for topic in topic_words:
                    score = 0
                    count = 0
                    for word1 in topic:
                        for word2 in topic:
                            if word1 == word2: continue
                            if word1 not in glove or word2 not in glove: continue
                            score += glove.similarity(word1, word2)
                            count += 1

                    results[metric]['c_we_per_topic'].append(0 if count == 0 else score / count)
                results[metric]['c_we'] = np.mean(results[metric]['c_we_per_topic'])
                results[metric]['c_we_std'] = np.std(results[metric]['c_we_per_topic'])

            while True:
                try:
                    self.log.debug('creating coherence model')
                    coherence_model = gensim.models.coherencemodel.CoherenceModel(topics=topic_words, texts=text,
                                                                                dictionary=dictionary,
                                                                                coherence=metric)
                    coherence_per_topic = coherence_model.get_coherence_per_topic()

                    topic_coherence = [coherence_per_topic[i] for i, t in enumerate(self.topics)]

                    results[metric][metric + '_per_topic'] = topic_coherence
                    results[metric][metric] = np.nanmean(coherence_per_topic)
                    results[metric][metric + '_std'] = np.nanstd(coherence_per_topic)

                    break

                except KeyError as e:
                    key = str(e)[1:-1]
                    for x in topic_words:
                        if key in x:
                            x.remove(key)

        return results
    
