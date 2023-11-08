import sys, os, json, time
import re, nltk
import argparse
from nltk import pos_tag
import pandas as pd
import numpy as np
from numpy import random
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
from models.lftm import LFTMwrapper

def combine(params):
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
        '0.99': None, 
    },
    'word_topic_pvalues': dict(),
    'coherence_metrics': defaultdict()
}


lda_param={'num_topics': [7, 10, 13], 'decay': [0.5,0.75,0.9], 'passes': [1,2,5]}

gsdmm_param={'K': [6, 8, 12], 'alpha': [0.1, 0.2], 'beta': [0.1, 0.2], 'n_iters': [1, 5, 15]}

lftm_param={'num_topics': [7, 10], 'alpha': [0.075,0.15,0.225], 'beta': [0.075,0.15,0.225], '_lambda': [0.6,0.75,0.9]}


def min_max(x, min, max): return min if x < min else max if x > max else x

def gen_kappa (): return min_max(abs(random.normal(loc=1, scale=0.1)), 1, 1.2)
def gen_K (): return min_max(int(abs(random.normal(loc=15, scale=7.5))), 10, 30)
def gen_T (): return min_max(int(abs(random.normal(loc=150, scale=35))), 100, 200)
def gen_alpha_gamma (): return min_max(abs(random.normal(loc=1, scale=0.05)), 0.85, 1.15)
def gen_eta (): return min_max(abs(random.normal(loc=0.01, scale=0.025)), 0.01, 0.12)

def gen_hdp_param(): return {'kappa': gen_kappa(), 'K': gen_K(), 'T': gen_T(),
           'alpha': gen_alpha_gamma(), 'gamma': gen_alpha_gamma(), 'eta': gen_eta()}

def gen_nb_t(): return random.randint(7,12)
def alpha_beta (): return min_max(random.normal(loc=0.15, scale=0.075), 0.05, 0.25)
def _lambda (): return min_max(random.normal(loc=0.75, scale=0.1), 0.6, 0.9)
def gen_lftm_param (): return {'num_topics': gen_nb_t(), 'alpha': alpha_beta(), 'beta': alpha_beta(), '_lambda': _lambda()}

data = preprocess_for_bow('data.csv')

def lda_experiment(data):
    experiment=dict()
    i=0
    print('running lda experiment')
    print(f"{len(combine(lda_param))} experimentations to make")
    for params in combine(lda_param):
        results = experiment_result.copy()
        param_set = dict(zip(list(lda_param.keys()), params))
        results['number_topics']=param_set['num_topics']
        results['hyperparameters']=param_set
        model=LDAwrappers(data['corpus'], data['dictionary'], 'LdaModelGensim', num_topics=param_set['num_topics'],
                        decay=param_set['decay'], passes=param_set['passes'])
        ntopics=len(model.topics())
        for pvalue in results['doc_topic_pvalues'].keys():
            idx=model.get_indexes_per_topics(data['corpus'], float(pvalue), data['ids'])
            results['doc_topic_pvalues'][pvalue]=[len(idx[str(i)])  if str(i) in idx.keys() else 0 for i in range(ntopics)]
        results['word_topic_pvalues']=model.topics(topn=10)
        results['coherence_metrics']=model.coherence(data['tokenized_data'], ['u_mass', 'c_we']) 
        experiment['exp_'+str(i)]=results
        i+=1
        if i%10==0:
            print(f"runnin the {i}th experiment")
    return experiment


def hdp_experiment(data, n=35):
    experiment=dict()
    print('running hdp experiment')
    print(f"{n} experimentations to make")
    for i in range(n):
        results = experiment_result.copy()
        param_set = gen_hdp_param()
        results['number_topics']=None
        results['hyperparameters']=param_set
        model=HDPwrapper(data['corpus'], data['dictionary'], kappa=param_set['kappa'], K=param_set['K'], T=param_set['T'], 
                            alpha=param_set['alpha'], gamma=param_set['gamma'], eta=param_set['eta'])
        ntopics=len(model.topics())
        for pvalue in results['doc_topic_pvalues'].keys():
            idx=model.get_indexes_per_topics(data['corpus'], float(pvalue), data['ids'])
            results['doc_topic_pvalues'][pvalue]=[len(idx[str(i)])  if str(i) in idx.keys() else 0 for i in range(ntopics)]
        results['word_topic_pvalues']=model.topics(topn=10)
        results['coherence_metrics']=model.coherence(data['tokenized_data'], ['u_mass', 'c_we']) 
        experiment['exp_'+str(i)]=results
        i+=1
        if i%1==0:
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
        ntopics=len(model.topics())
        for pvalue in results['doc_topic_pvalues'].keys():
            idx=model.get_indexes_per_topics(data['corpus'], float(pvalue), data['ids'])
            results['doc_topic_pvalues'][pvalue]=[len(idx[str(i)])  if str(i) in idx.keys() else 0 for i in range(ntopics)]
        results['word_topic_pvalues']=model.topics(topn=10)
        results['coherence_metrics']=model.coherence(data['tokenized_data'], ['u_mass', 'c_we']) 
        experiment['exp_'+str(i)]=results
        i+=1
        if i%5==0:
            print(f"runnin the {i}th experiment")
    return experiment


def lftm_experiment(datapath, n=20):
    experiment=dict()
    i=0
    print('running lftm experiment')
    print(f"{n} experimentations to make")
    for i in range(n):
        results = experiment_result.copy()
        param_set = gen_lftm_param()
        results['number_topics']=param_set['num_topics']
        results['hyperparameters']=param_set
        model=LFTMwrapper(datapath, num_topics=param_set['num_topics'], alpha=param_set['alpha'], 
                        beta=param_set['beta'], _lambda=param_set['_lambda'])
        ntopics=len(model.topics())
        #for pvalue in results['doc_topic_pvalues'].keys():
            #idx=model.get_indexes_per_topics(datapath, float(pvalue))
            #results['doc_topic_pvalues'][pvalue]=[len(idx[str(i)])  if str(i) in idx.keys() else 0 for i in range(ntopics)]
        results['word_topic_pvalues']=model.topics()
        results['coherence_metrics']=model.coherence(data['tokenized_data'], ['u_mass', 'c_we']) 
        experiment['exp_'+str(i)]=results
        model.clean_dir()
        i+=1
        if i%1==0:
            print(f"runnin the {i}th experiment.")
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
            #gsdmm_exp = gsdmm_experiment(data)
            lftm_exp = lftm_experiment(args.data_path)
            #end job
            #experiment_results['lda_experiment']=lda_exp
            #experiment_results['hdp_experiment']=hdp_exp
            #experiment_results['gsdmm_experiment']=gsdmm_exp
            experiment_results['lftm_experiment']=lftm_exp
            save_json(args.experiment_file, experiment_results)
    else:
        #job
        experiment_results=dict()
        lda_exp = lda_experiment(data)
        hdp_exp = hdp_experiment(data)
        gsdmm_exp = gsdmm_experiment(data)
        lftm_exp = lftm_experiment(args.data_path)
        #end job
        experiment_results['lda_experiment']=lda_exp
        experiment_results['hdp_experiment']=hdp_exp
        experiment_results['gsdmm_experiment']=gsdmm_exp
        experiment_results['lftm_experiment']=lftm_exp
        save_json(args.experiment_file, experiment_results)


if __name__ == "__main__":
    main()




        