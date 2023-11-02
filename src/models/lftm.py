from .abstract_model import AbstractModel
import numpy as np
import os, nltk, pickle, subprocess
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

class LFTMwrapper(AbstractModel):
    """Latent Feature Topic Model

    Source: https://github.com/datquocnguyen/LFTM

    NOTE: all adj/noun pairs and unicodes will be removed 
                        -> need to change preprocessing/lftm implementation to fix this
    """
    def __init__(self, text, #tokenized text
                    id2word, glove_path='glove/glove.6B.300d.txt', #check might need 50d
                    lftm_jar='LFTM.jar', num_topics=10, alpha=0.1, 
                    beta=0.1,
                    _lambda=1,
                    initer=50,
                    niter=5,
                    twords=10,
                    model='LFLDA',
                    name='test'): 
        super().__init__(None, None, None) 
        if model not in ['LFLDA', 'LFDMM']:
            raise ValueError('Model should be LFLDA (default) or LFDMM.')
        self.text = text
        self.id2word=id2word

        #glove
        if os.path.exists(glove_path)==False:
            raise ValueError('glove_path does not contain the data')
        else:
            w2v = utils.get_tmpfile("w2v")
            glove2word2vec(glove_path, w2v)
            glove = KeyedVectors.load_word2vec_format(w2v)

        corpus_vocab = list(self.id2word.values())
        tok2remove = {}
        for t in corpus_vocab:
            if t not in glove:
                tok2remove[t] = True

        self.text = [remove_tokens(doc, tok2remove) for doc in self.text]

        with open('datalftm.txt', 'w') as fout:
            for i in self.text:
                fout.write(i+'\n')

        binary_path = os.path.dirname(os.path.abspath(__file__))+'/'+ lftm_jar
        proc = f'java -jar {binary_path} -model {model} -corpus datalftm.txt -vectors {glove_path} -ntopics {num_topics} ' \
               f'-alpha {alpha} -beta {beta} -lambda {_lambda} -initers {initer} -niters {niter} -twords {twords} ' \
               f'-name {name} -sstep 0'
        print(proc)
        self.log.debug('Executing: ' + proc)

        logWrap = LoggerWrapper(self.log)

        completed_proc = subprocess.run(proc, shell=True, stdout=logWrap, stderr=logWrap)
        self.log.debug(f'Completed with code {completed_proc.returncode}')

        if completed_proc.returncode == 0:
            self.training_success = True
        else:
            self.training_success = False

