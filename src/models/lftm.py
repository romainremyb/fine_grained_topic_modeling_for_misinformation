from .abstract_model import AbstractModel
import numpy as np
import os, nltk, pickle, subprocess, re
import gensim
from gensim.models import KeyedVectors
from gensim.test import utils
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import LdaModel, LdaMulticore, HdpModel
import logging
from collections import defaultdict
from utils import preprocess_for_bow, preprocess, LoggerWrapper

def remove_tokens(x, tok2remove):
    return ' '.join(['' if t in tok2remove else t for t in x])

TOPIC_REGEX = r'Topic(\d+): (.+)'


class LFTMwrapper(AbstractModel):
    """Latent Feature Topic Model

    Source: https://github.com/datquocnguyen/LFTM

    NOTE: all adj/noun pairs and unicodes will be removed 
                        -> need to change preprocessing/lftm implementation to fix this

    - LFLDA.theta: 
    - LFLDA.phi:
    - LFLDA.topWords:
    - LFLDA.topicAssignments: 
    - LFLDA.paras: 
    """
    def __init__(self, datasetpath, glove_path='glove/glove.6B.300d.txt', #check might need 50d
                    lftm_jar='LFTM.jar', num_topics=10, alpha=0.1, 
                    beta=0.1,
                    _lambda=1,
                    initer=50,
                    niter=5,
                    twords=10,
                    model='LFLDA',
                    name='test'): 
        super().__init__(None, None, None) 

        if os.path.exists(datasetpath)==False:
            raise ValueError('wrong datapath')
        else:
            data = preprocess_for_bow(datasetpath)

        if model not in ['LFLDA', 'LFDMM']:
            raise ValueError('Model should be LFLDA (default) or LFDMM.')
            
        self.text = data['tokenized_data']
        self.id2word=data['dictionary']
        self.ids=data['ids']
        self.model=model
        self.glove_path=glove_path
        self.initer=initer
        self.niter=niter
        self.twords=twords
        self.name=name

        self.top_words = None
        self.paras_path = None
        self.theta_path_model = None
        self.topicAssignments = None
        self.theta = None
        path = os.getcwd()
        self.update_model_path(path, name)

        #glove
        if os.path.exists(glove_path)==False:
            raise ValueError('glove_path does not contain the data')
        else:
            w2v = utils.get_tmpfile("w2v")
            glove2word2vec(glove_path, w2v)
            self.glove = KeyedVectors.load_word2vec_format(w2v)

        corpus_vocab = list(self.id2word.values())
        tok2remove = {}
        for t in corpus_vocab:
            if t not in self.glove:
                tok2remove[t] = True

        self.text = [remove_tokens(doc, tok2remove) for doc in self.text]

        with open('datalftm.txt', 'w') as fout:
            for i in self.text:
                fout.write(i+'\n')

        self.binary_path = os.path.dirname(os.path.abspath(__file__))+'/'+ lftm_jar
        proc = f'java -jar {self.binary_path} -model {self.model} -corpus datalftm.txt -vectors {self.glove_path} -ntopics {num_topics} ' \
               f'-alpha {alpha} -beta {beta} -lambda {_lambda} -initers {initer} -niters {niter} -twords {twords} ' \
               f'-name {self.name} -sstep 0'

        self.log.debug('Executing: ' + proc)

        logWrap = LoggerWrapper(self.log)

        completed_proc = subprocess.run(proc, shell=True, stdout=logWrap, stderr=logWrap)
        self.log.debug(f'Completed with code {completed_proc.returncode}')

        if completed_proc.returncode == 0:
            self.training_success = True
        else:
            self.training_success = False
            

    def update_model_path(self, model_root, name):
        model_root = os.path.abspath(model_root)
        self.model_path = model_root
        self.top_words = model_root + '/%s.topWords' % name
        self.paras_path = model_root + '/%s.paras' % name
        self.theta_path = model_root + '/%s.theta' % name
        self.phi_path = model_root + '/%s.phi' % name
        self.topic_assignments = model_root + '/%s.topicAssignments' % name
        self.data_glove = model_root + '/%s.glove' % name


    def clean_dir(self):  #TODO ? put these file to memory and delete in main ?
        for file in [self.top_words, self.paras_path, self.theta_path, self.topic_assignments, self.phi_path]:
                os.remove(file)


    def get_document_topics(self, documents, minimum_probability=0.5):
        """documents: path or list of tokenized doc data
        """
        if type(documents)==str:
            if os.path.exists(documents)==False:
                raise ValueError('wrong datapath')
            else:
                data = preprocess_for_bow(documents)
                documents = data['tokenized_data']

        corpus_vocab = list(self.id2word.values())
        tok2remove = {}
        for t in corpus_vocab:
            if t not in self.glove:
                tok2remove[t] = True

        text = [remove_tokens(doc, tok2remove) for doc in documents]

        with open('inference_docs.txt', 'w') as fout:
            for i in text:
                fout.write(i+'\n')

        proc = f'java -jar {self.binary_path} -model {self.model}inf -paras {self.paras_path} -corpus inference_docs.txt ' \
               f'-initers {self.initer} -niters {self.niter} -twords {self.twords} -name {self.name}inf -sstep 0'

        logWrap = LoggerWrapper(self.log)
        completed_proc = subprocess.run(proc, shell=True, stderr=logWrap, stdout=logWrap)
        self.log.debug(f'Completed with code {completed_proc.returncode}')

        with open(os.path.join(os.getcwd(),self.name+'inf.theta'), "r") as file:
            doc_topic_dist = [line.strip().split() for line in file.readlines()]
        
        topics = [[(i, float(score)) for i, score in enumerate(doc) if float(score)>=minimum_probability]
                  for doc in doc_topic_dist]
        
        return topics


    def get_indexes_per_topics(self, docs, minimum_probability):
        result=defaultdict(list)
        generator = self.get_document_topics(docs, minimum_probability=minimum_probability)
        for i in range(len(generator)):
            for topic in generator[i]:
                result[str(topic[0])].append(self.ids[i])
        return result


    def get_corpus_predictions(self, minimum_probability=0.5):
        with open(self.theta_path_model, "r") as file:
            doc_topic_dist = [line.strip().split() for line in file.readlines()]

        topics = [[(i, float(score)) for i, score in enumerate(doc) if float(score)>=minimum_probability]
                  for doc in doc_topic_dist]

        return topics


    def topics(self):
        topics = []
        with open(self.top_words, 'r') as f:
            for line in f:
                match = re.match(TOPIC_REGEX, line.strip())
                if not match:
                    continue
                _id, words = match.groups()
                topics.append({'words': words.split()})

        return topics

    