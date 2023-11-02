import re
import nltk
from nltk import pos_tag
import sys
import spacy
from gensim.corpora import Dictionary
import numpy as np
from collections import defaultdict


init_done = False

unicodes_to_remove = ['\\u2015', '\\u2033', '\\u2063', '\\u0627',
 '\\u2066', '\\u2069', '\\u200d', '\\u2013', '\\ufe0f', '\\u2026',
 '\\u2018', '\\u2014', '\\u2022', '\\u201d', '\\u201c', '\\u2019']


experiment_result = {
    'number_topics': None, #int, None if infered ex: HDP
    'hyperparameters': {},
    'doc_topic_pvalues': { # get for all above or filter in range ?
        '0.35': {}, #list of ids
        '0.50': {}, #list of ids
        '0.60': {}, 
        '0.75': {}, 
        '0.90': {}, 
        '0.95': {}, 
        '0.95': {}, 
    },
    'word_topic_pvalues': dict(),
    'coherence_metrics': defaultdict()
}


def is_list_of_strings(lst):
    return bool(lst) and isinstance(lst, list) and all(isinstance(elem, str) for elem in lst)


#TODO: add wordName/id dictionary ?
def preprocess_for_bow(data, return_idxs=True, preprocessing=True, preproc_params=
                       {'keep_unicodes': {'keep': True, 'min_count_in_corpus': 2},
                        'strip_brackets': False, 'add_adj_nn_pairs': True, 'verbs': True, 
                        'adjectives': False}): 
    """
    :return: Dict with keys: 
        - data as list of text docs
        - ids if flagged
        - data as list of documents - made up of list of tokens
        - dictionary of tokens
        - bag of words (ids pointing to dictionary)

    :param data: .csv | .txt path (.txt treating lines as text doc only) | list of docs
    :param bool return_idxs: whether to print indexes or not (note empty docs are removed)
    :param bool preprocessing
    :param dict preproc_params
        preproc_params.keep_unicodes.keep: set True to keep them in BOW - modifying unicodes_to_remove is recommanded
        preproc_params.keep_unicodes.min_count_in_corpus: checks counts of unicodes in corpus and 
                            remove unicodes with count < min_count_in_corpus 
                            (e.g. setting as 2 will remove unicodes of frequency one)
        preproc_params.strip_brackets: remove from bow what is inside brackets
        preproc_params.add_adj_nn_pairs: whether to add adj/noun pairs to BOW (keeps them as sole items)
        preproc_params.verbs: whether to keep verbs (identified by pos) in bow
        add_adj_nn_pairs.adjectives: whether to keep adjectives (identified by pos) in bow
            note: adverbs, prepositions etc not retrieved
    """

    finaldata=dict()
    if type(data) == str:
        try:
            if data[-4:]=='.csv':
                with open(data, "r", encoding='utf-8') as datafile:
                    text, idxs = [], []
                    for line in datafile:
                        if len(line.rstrip().split('",',1)[-1][1:-1])>=3:
                            text.append(line.rstrip().split('",',1)[-1][1:-1])
                            idxs.append(line.rstrip().split('",',1)[0][1:-1])
            elif data[-4:]=='.txt': #TODO -> try with .txt file
                return_idxs=False #TODO: Log the fact that is is considering entire lines as text docs
                with open(data, "r", encoding='utf-8') as datafile:
                    text = [line.rstrip() for line in datafile if line and len(line)!=0]
            else:
                print('file neither .csv nor .txt')
                return

        except FileNotFoundError as e:
            print(f"path {data} not found !", file=sys.stderr)
            return

    elif is_list_of_strings(data):
        text = data
    
    elif isinstance(data,str)==True:
        text=[data]
        
    else:
        raise ValueError('data should be a path or a list of strings')
    
    if preprocessing:
        if preproc_params['keep_unicodes']['keep']==True:
            unicodes_to_remove.extend(remove_unicodes_with_min_count(text, 
                                preproc_params['keep_unicodes']['min_count_in_corpus']))
            
        text = [preprocess(doc, preproc_params['strip_brackets'], preproc_params['keep_unicodes']['keep'], 
                           preproc_params['add_adj_nn_pairs'],  preproc_params['verbs'], 
                           preproc_params['adjectives'], unicodes_to_remove) for doc in text]
    
        if return_idxs==True:
            try:
                tokenized_data, dictionary, corpus = return_data(text[1:])
                finaldata['text'], finaldata['ids'] = text[1:], idxs[1:]
                finaldata['tokenized_data'], finaldata['dictionary'], finaldata['corpus'] = tokenized_data, dictionary, corpus 
                return finaldata
            except NameError:
                print('idxs list not defined')
        else:
            tokenized_data, dictionary, corpus = return_data(text[1:])
            finaldata['text'] = text[1:]
            finaldata['tokenized_data'], finaldata['dictionary'], finaldata['corpus'] = tokenized_data, dictionary, corpus 
            return finaldata
        
    else:
        finaldata['text'] = text
        return finaldata



def return_data(text_list):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokenized_data = [[token.strip() for token in tokenizer.tokenize(text)] for text in text_list]
    dictionary = Dictionary(documents=tokenized_data, prune_at=None)
    corpus = [dictionary.doc2bow(seq) for seq in tokenized_data]
    return tokenized_data, dictionary, corpus

def _init():
    global init_done
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    init_done = True

def remove_unicodes_with_min_count(text_list, min_count):
    unicodes=[]
    to_remove=[]
    pattern1 = r'\\U[0-9A-Za-z]{8}'
    pattern2 = r'\\u[0-9A-Za-z]{4}'
    for text in text_list:
        filtered_text = re.findall(pattern1, text.encode('unicode-escape').decode())
        unicodes.extend([u for u in filtered_text])
        filtered_text = re.findall(pattern2, text.encode('unicode-escape').decode())
        unicodes.extend([u for u in filtered_text])
    for i in range(min_count+1):
        idxs=np.where(np.unique(unicodes, return_counts=True)[1]==i)
        to_remove.extend(list(np.unique(unicodes, return_counts=True)[0][idxs]))
    return list(set(to_remove)) #make unique

def preprocess(text, strip_brackets=False, keep_unicodes=True, add_adj_nn_pairs=True,  verbs=True, adjectives=False, unicodes_to_remove=unicodes_to_remove):

    global init_done

    if not init_done:
        _init()

    if strip_brackets:
        text = re.sub(r'\((.*?)\)', ' ', text)

    if keep_unicodes:
        pattern1 = r'\\U[0-9A-Za-z]{8}'
        pattern2 = r'\\u[0-9A-Za-z]{4}'
        unicodes = re.findall(pattern1, text.encode('unicode-escape').decode())
        unicodes.extend(re.findall(pattern2, text.encode('unicode-escape').decode()))
        unicodes = [u for u in unicodes if u not in unicodes_to_remove]
    
    text = re.sub(r'\b\d+\b', '', text) #or all digits ex: r'\d+'

    text = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text.lower()) # tokenize space seperated group of characters

    lem = nltk.stem.WordNetLemmatizer()

    if add_adj_nn_pairs:
        adj_noun_pairs = add_JJ_NN_pairs_nltk(text, lem) #merge adjectives ?

    text = pos_and_lemma(text, lem, verbs, adjectives) 

    text = [w for w in text if w not in nltk.corpus.stopwords.words('english')]

    text = [w for w in text if len(w) >= 3] #ignore with two letters or less

    if add_JJ_NN_pairs_nltk:
        text.extend(adj_noun_pairs)

    if keep_unicodes:
        text.extend(unicodes)
    
    text = ' '.join(text)
    return text


def pos_and_lemma(tokens, lem, verbs, adjectives):
    # nltk pos tags: https://www.guru99.com/pos-tagging-chunking-nltk.html
    # lem.lemmatize(i,j) -> j: POS tag. Valid options are `"n"` for nouns,`"v"` for verbs, `"a"` for adjectives, `"r"` for adverbs and `"s"` for satellite adjectives.
    result=[]
    if verbs==True:
        check=['n','v']
    else:
        check=['n']
    for i, j in pos_tag(tokens):
        if j[0].lower() in check:
            result.append(lem.lemmatize(i, j[0].lower()))
        if adjectives==True and j[0].lower()=='j':
            result.append(lem.lemmatize(i, 'a'))
    return result

def add_JJ_NN_pairs_nltk(tokens, lem):
    adj_noun_pairs = []
    pos_tagged = pos_tag(tokens)
    for i in range(len(pos_tagged) - 1):
        word, pos = pos_tagged[i]
        next_word, next_pos = pos_tagged[i + 1]
        if pos.startswith('JJ') and next_pos.startswith('NN'):
            adj_noun_pairs.append(lem.lemmatize(word,'a') + lem.lemmatize(next_word,'a'))
    return adj_noun_pairs

def add_JJ_NN_pairs_spacy(text, lem):
    adj_noun_pairs = []
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
            adj_noun_pairs.append(lem.lemmatize(token.text ,'a') + lem.lemmatize(token.head.text,'a'))
    return adj_noun_pairs


# source https://codereview.stackexchange.com/questions/6567/redirecting-subprocesses-output-stdout-and-stderr-to-the-logging-module

import os
import logging
import threading


class LoggerWrapper(threading.Thread):
    """
    Read text message from a pipe and redirect them
    to a logger (see python's logger module),
    the object itself is able to supply a file
    descriptor to be used for writing

    fdWrite ==> fdRead ==> pipeReader
    """

    def __init__(self, logger, level=logging.DEBUG):
        """
        Setup the object with a logger and a loglevel
        and start the thread
        """

        # Initialize the superclass
        threading.Thread.__init__(self)

        # Make the thread a Daemon Thread (program will exit when only daemon
        # threads are alive)
        self.daemon = True

        # Set the logger object where messages will be redirected
        self.logger = logger

        # Set the log level
        self.level = level

        # Create the pipe and store read and write file descriptors
        self.fdRead, self.fdWrite = os.pipe()

        # Create a file-like wrapper around the read file descriptor
        # of the pipe, this has been done to simplify read operations
        self.pipeReader = os.fdopen(self.fdRead)

        # Start the thread
        self.start()

    # end __init__

    def fileno(self):
        """
        Return the write file descriptor of the pipe
        """
        return self.fdWrite

    # end fileno

    def run(self):
        """
        This is the method executed by the thread, it
        simply read from the pipe (using a file-like
        wrapper) and write the text to log.
        NB the trailing newline character of the string
           read from the pipe is removed
        """

        # Endless loop, the method will exit this loop only
        # when the pipe is close that is when a call to
        # self.pipeReader.readline() returns an empty string
        while True:

            # Read a line of text from the pipe
            messageFromPipe = self.pipeReader.readline()

            # If the line read is empty the pipe has been
            # closed, do a cleanup and exit
            # WARNING: I don't know if this method is correct,
            #          further study needed
            if len(messageFromPipe) == 0:
                self.pipeReader.close()
                os.close(self.fdRead)
                return
            # end if

            # Remove the trailing newline character frm the string
            # before sending it to the logger
            if messageFromPipe[-1] == os.linesep:
                messageToLog = messageFromPipe[:-1]
            else:
                messageToLog = messageFromPipe
            # end if

            # Send the text to the logger
            self._write(messageToLog)
        # end while

    # end run

    def _write(self, message):
        """
        Utility method to send the message
        to the logger with the correct loglevel
        """
        self.logger.log(self.level, message)
    # end write
