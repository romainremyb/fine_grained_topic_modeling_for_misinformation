from .abstract_model import AbstractModel
import numpy as np
import os
from gensim.models import LdaModel, LdaMulticore, HdpModel
import logging
from collections import defaultdict

ROOT = '.'

class HDPwrapper(AbstractModel):
    def __init__(self, bow_corpus, id2word, overchunking_factor=1): #TODO check the overchunking_factor -> setting as 256 should give simiular results as with None -> check with a range of factors from (1 to 256)
        super().__init__(bow_corpus, id2word, None)
        #https://radimrehurek.com/gensim/models/hdpmodel.html
        self.model = HdpModel(bow_corpus, id2word,
                max_chunks=overchunking_factor*len(bow_corpus), #Upper bound on how many chunks to process (retakes chunks from beginning of corpus if not enough docs)
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
        
