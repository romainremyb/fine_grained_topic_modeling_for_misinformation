from .abstract_model import AbstractModel
import numpy as np
import os, nltk
from collections import defaultdict
from utils import preprocess_for_bow, preprocess

from contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

class CTMwrapper(AbstractModel):
    def __init__(self, data_path, num_topics = 10, preprocessing=True, preproc_params = 
                        {'keep_unicodes': {'keep': True, 'min_count_in_corpus': 2}, 'strip_brackets': False, 
                            'add_adj_nn_pairs': True, 'verbs': True, 'adjectives': False},
                    bert_model="paraphrase-distilroberta-base-v2",
                    contextual_size=768,
                    num_epochs=1,
                    hidden_sizes=(100,),
                    batch_size=200,
                    inference_type="combined",
                    model_type='prodLDA',
                    activation='softplus',
                    dropout=0.2,
                    learn_priors=True,
                    lr=2e-3,
                    momentum=0.99,
                    solver='adam',
                    reduce_on_plateau=False):
        
        super().__init__(None, None, None)

        data = preprocess_for_bow(data_path, preprocessing=preprocessing, preproc_params = preproc_params)
        self.raw_text = preprocess_for_bow(data_path, preprocessing=False)['text'][1:] #NOTE: LM embedding takes raw text -> change if disire custum
        self.text_for_bow = data['text']
        self.ids = data['ids'][1:]
        self.tokenized_data = data['tokenized_data']
        self.dictionary = data['dictionary']

        self.qt = TopicModelDataPreparation(bert_model)
        training_dataset = self.qt.fit(text_for_contextual=self.raw_text, text_for_bow=self.text_for_bow)

        if inference_type=="combined":
            self.model = CombinedTM(bow_size=len(self.dictionary), contextual_size=contextual_size, num_epochs=num_epochs,
                        model_type=model_type, hidden_sizes=hidden_sizes, activation=activation,
                        dropout=dropout, learn_priors=learn_priors, lr=lr, momentum=momentum,
                        solver=solver, reduce_on_plateau=reduce_on_plateau, n_components=num_topics,
                        batch_size=batch_size)
        else:
            #TODO add other inference type e.g. zero-shot
            raise ValueError("Wrong inference type")
        
        self.model.fit(training_dataset)
        self.corpus_predictions = self.model.get_thetas(training_dataset)