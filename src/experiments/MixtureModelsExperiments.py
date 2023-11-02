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
from utils import preprocess_for_bow
from models.lda import LDAwrappers
from models.hdp import HDPwrapper
from models.gsdmm import MovieGroupProcessWrapper

def combine(params):
    param_names = list(params.keys())
    param_values = list(params.values())
    param_combinations = list(itertools.product(*param_values))
    return param_combinations

experiment_result = {
    'number_topics': None, #int, None if infered ex: HDP
    'hyperparameters': {},
    'doc_topic_pvalues': { # get for all above or filter in range ?
        '0.35': None, #key: topic -> list of ids
        '0.50': None, 
        '0.60': None, 
        '0.75': None, 
        '0.90': None, 
        '0.95': None, 
        '0.95': None, 
    },
    'word_topic_pvalues': dict(),
    'coherence_metrics': defaultdict()
}


lda_param={'num_topics': [i for i in range(5,20) if i%2==0], 'decay': [0.5,0.75,0.9], 'passes': [1,2,5]}

hdp_param={'kappa': [1,1.05,1.1], 'K': [10, 15, 30], 'T': [100,150,200],
           'alpha': [0.9,1,1.1], 'gamma': [0.9,1,1.1], 'eta': [0.01, 0.05]}

gsdmm_param={'K': [i for i in range(6,27) if i%3==0], 'alpha': [0.1, 0.2], 'beta': [0.1, 0.2], 'n_iters': [1, 5, 15]}

data = preprocess_for_bow('data.csv')

def lda_experiment(data):
    experiment=dict()
    i=0
    print('running lda experiment')
    print(f"{len(combine(lda_param))} experimentations to make")
    for params in combine(lda_param):
        results = experiment_result.copy()
        param_set = dict(zip(list(lda_param.keys()), params))
        print(param_set)
        results['number_topics']=param_set['num_topics']
        results['hyperparameters']=param_set
        model=LDAwrappers(data['corpus'], data['dictionary'], 'LdaModelGensim', num_topics=param_set['num_topics'],
                        decay=param_set['decay'], passes=param_set['passes'])
        for pvalue in results['doc_topic_pvalues'].keys():
            results['doc_topic_pvalues'][pvalue]=len(model.get_indexes_per_topics(data['corpus'], float(pvalue), data['ids']))
        results['word_topic_pvalues']=model.topics(topn=10)
        results['coherence_metrics']=model.coherence(data['tokenized_data'], ['u_mass', 'c_we']) 
        experiment['exp_'+str(i)]=results
        i+=1
        if i%10==0:
            print(f"runnin the {i}th experiment")
    return experiment


def hdp_experiment(data):
    experiment=dict()
    i=0
    print('running hdp experiment')
    print(f"{len(combine(hdp_param))} experimentations to make")
    for params in combine(hdp_param):
        results = experiment_result.copy()
        param_set = dict(zip(list(hdp_param.keys()), params))
        results['number_topics']=None
        results['hyperparameters']=param_set
        model=HDPwrapper(data['corpus'], data['dictionary'], kappa=param_set['kappa'], K=param_set['K'], T=param_set['T'], 
                            alpha=param_set['alpha'], gamma=param_set['gamma'], eta=param_set['eta'])
        for pvalue in results['doc_topic_pvalues']:
            results['doc_topic_pvalues'][pvalue]=len(model.get_indexes_per_topics(data['corpus'], float(pvalue), data['ids']))
        results['word_topic_pvalues']=model.topics(topn=10)
        results['coherence_metrics']=model.coherence(data['tokenized_data'], ['u_mass', 'c_we']) 
        experiment['exp_'+str(i)]=results
        i+=1
        if i%5==0:
            print(f"running the {i}th experiment")
    return experiment


def gsdmm_experiment(data):
    experiment=dict()
    i=0
    print('running gcdmm experiment')
    print(f"{len(combine(gsdmm_param))} experimentations to make")
    for params in combine(gsdmm_param):
        results = experiment_result.copy()
        param_set = dict(zip(list(gsdmm_param.keys()), params))
        results['number_topics']=param_set['K']
        results['hyperparameters']=param_set
        model=MovieGroupProcessWrapper(data['corpus'], data['dictionary'], K=param_set['K'], 
                    alpha=param_set['alpha'], beta=param_set['beta'], n_iters=param_set['n_iters'])
        for pvalue in results['doc_topic_pvalues']:
            results['doc_topic_pvalues'][pvalue]=len(model.get_indexes_per_topics(data['corpus'], float(pvalue), data['ids']))
        results['word_topic_pvalues']=model.topics(topn=10)
        results['coherence_metrics']=model.coherence(data['tokenized_data'], ['u_mass', 'c_we']) 
        experiment['exp_'+str(i)]=results
        i+=1
        if i%5==0:
            print(f"runnin the {i}th experiment")
    return experiment


def save_json(path, dictionary):
    with open(path, 'w') as json_file:
        json.dump(dictionary, json_file)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='dataset path')
    parser.add_argument('--experiment_file', type=str, required=True, help='experiment file path')
    args = parser.parse_args()
    data = preprocess_for_bow(args.data_path)

    if os.path.exists(args.experiment_file):
        with open(args.experiment_file, 'r') as json_file:
            experiment_results = json.load(json_file)
            #job
            #lda_exp = lda_experiment(data)
            #hdp_exp = hdp_experiment(data)
            gsdmm_exp = gsdmm_experiment(data)
            #end job
            #experiment_results['lda_experiment']=lda_exp
            #experiment_results['hdp_experiment']=hdp_exp
            experiment_results['gsdmm_experiment']=gsdmm_exp
            save_json(args.experiment_file, experiment_results)
    else:
        #job
        experiment_results=dict()
        lda_exp = lda_experiment()
        hdp_exp = hdp_experiment()
        gsdmm_exp = gsdmm_experiment()
        #end job
        experiment_results['lda_experiment']=lda_exp
        experiment_results['hdp_experiment']=hdp_exp
        experiment_results['gsdmm_experiment']=gsdmm_exp
        save_json(args.experiment_file, experiment_results)


if __name__ == "__main__":
    main()




        