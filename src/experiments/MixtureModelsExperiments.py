import sys, os, json
import re, nltk
import argparse
from nltk import pos_tag
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn, gensim
from sklearn.decomposition import PCA
import itertools
from gensim.corpora import Dictionary
from collections import defaultdict
sys.path.append(os.path.expanduser("~")+'/Desktop/topic_modeling/fine_grained_topic_modeling_for_misinformation/src/')
sys.path.append(os.path.expanduser("~")+'/Desktop/topic_modeling/fine_grained_topic_modeling_for_misinformation/src/')
os.chdir(os.path.expanduser("~")+'/Desktop/topic_modeling/fine_grained_topic_modeling_for_misinformation/data/')
from utils import preprocess_for_bow, experiment_result
from models.lda import LDAwrappers
from models.hdp import HDPwrapper
from models.gsdmm import MovieGroupProcessWrapper

def combine(params):
    param_names = list(params.keys())
    param_values = list(params.values())
    param_combinations = list(itertools.product(*param_values))
    return param_combinations


lda_param={'num_topics': range(5,20), 'decay': [0.5,0.75,0.9]}

hdp_param={'kappa': [1,1.05,1.1], 'K': [10, 15, 30], 'T': [100,150,200],
           'alpha': [0.9,1,1.1], 'gamma': [0.9,1,1.1], 'eta': [0.01, 0.05]}

gsdmm_param={'K': range(7,25), 'alpha': [0.1, 0.25], 'beta': [0.1, 0.25]}

data = preprocess_for_bow('data.csv')

def lda_experiment():
    experiment=dict()
    i=0
    print('running lda experiment')
    print(f"{len(combine(lda_param))} experimentations to make")
    for params in combine(lda_param):
        results = experiment_result.copy()
        param_set = dict(zip(list(lda_param.keys()), params))
        results['number_topics']=param_set['num_topics']
        results['hyperparameters']=param_set
        lda=LDAwrappers(data['corpus'], data['dictionary'], 'LdaModelGensim', num_topics=param_set['num_topics'],
                        decay=param_set['decay'])
        for pvalue in results['doc_topic_pvalues']:
            results['doc_topic_pvalues'][pvalue]=lda.get_indexes_per_topics(data['corpus'], pvalue, data['ids'])
        results['word_topic_pvalues']=lda.topics(topn=10)
        results['coherence_metrics']=lda.coherence(data['tokenized_data'], ['u_mass']) #c_we?
        experiment['exp_'+str(i)]=results
        i+=1
        if i%10==0:
            print(f"runnin the {i}th experiment")
    return experiment


def hdp_experiment():
    experiment=dict()
    i=0
    print('running hdp experiment')
    print(f"{len(combine(hdp_param))} experimentations to make")
    for params in combine(hdp_param):
        results = experiment_result.copy()
        param_set = dict(zip(list(hdp_param.keys()), params))
        results['number_topics']=param_set['num_topics']
        results['hyperparameters']=param_set
        model=HDPwrapper(data['corpus'], data['dictionary'], kappa=param_set['kappa'], K=param_set['K'], T=param_set['T'], 
                            alpha=param_set['alpha'], gamma=param_set['gamma'], eta=param_set['eta'])
        for pvalue in results['doc_topic_pvalues']:
            results['doc_topic_pvalues'][pvalue]=model.get_indexes_per_topics(data['corpus'], pvalue, data['ids'])
        results['word_topic_pvalues']=model.topics(topn=10)
        results['coherence_metrics']=model.coherence(data['tokenized_data'], ['u_mass']) #c_we?
        experiment['exp_'+str(i)]=results
        i+=1
        if i%10==0:
            print(f"runnin the {i}th experiment")
    return experiment


def gsdmm_experiment():
    experiment=dict()
    i=0
    print('running gcdmm experiment')
    print(f"{len(combine(gsdmm_param))} experimentations to make")
    for params in combine(gsdmm_param):
        results = experiment_result.copy()
        param_set = dict(zip(list(gsdmm_param.keys()), params))
        results['number_topics']=param_set['num_topics']
        results['hyperparameters']=param_set
        model=MovieGroupProcessWrapper(data['corpus'], data['dictionary'], )
        for pvalue in results['doc_topic_pvalues']:
            results['doc_topic_pvalues'][pvalue]=model.get_indexes_per_topics(data['corpus'], pvalue, data['ids'],
                            K=param_set['K'], alpha=param_set['alpha'], beta=param_set['beta'],)
        results['word_topic_pvalues']=model.topics(topn=10)
        results['coherence_metrics']=model.coherence(data['tokenized_data'], ['u_mass', 'c_we']) 
        experiment['exp_'+str(i)]=results
        i+=1
        if i%10==0:
            print(f"runnin the {i}th experiment")
    return experiment


def save_json(path, dictionary):
    with open(path, 'w') as json_file:
        json.dump(dictionary, json_file)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_file', type=str, required=True, help='experiment file path')
    args = parser.parse_args()
    if os.path.exists(args.experiment_file):
        with open(args.experiment_file, 'r') as json_file:
            experiment_results = json.load(json_file)
            #job
            lda_exp = lda_experiment()
            hdp_exp = hdp_experiment()
            gsdmm = gsdmm_experiment()
            #end job
            experiment_results['lda_experiment']=lda_exp
            experiment_results['hdp_experiment']=hdp_exp
            experiment_results['gsdmm_experiment']=gsdmm_exp
            save_json(args.experiment_file, experiment_results)
    else:
        #job
        lda_exp = lda_experiment()
        hdp_exp = hdp_experiment()
        gsdmm = gsdmm_experiment()
        #end job
        experiment_results['lda_experiment']=lda_exp
        experiment_results['hdp_experiment']=hdp_exp
        experiment_results['gsdmm_experiment']=gsdmm_exp
        save_json(args.experiment_file, experiment_results)


if __name__ == "__main__":
    main()




        