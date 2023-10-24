import re
import nltk
from nltk import pos_tag
import sys
import spacy


init_done = False

unicodes_to_remove = ['\\u2015', '\\u2033', '\\u2063', '\\u0627',
 '\\u2066', '\\u2069', '\\u200d', '\\u2013', '\\ufe0f', '\\u2026',
 '\\u2018', '\\u2014', '\\u2022', '\\u201d', '\\u201c', '\\u2019']


def is_list_of_strings(lst):
    return bool(lst) and isinstance(lst, list) and all(isinstance(elem, str) for elem in lst)


def preprocess_for_latent_distribs(data, preprocessing=False, return_idxs=False):
    if type(data) == str:
        try:
            if data[-4:]=='.csv':
                with open(data, "r", encoding='utf-8') as datafile:
                    text, idxs = [], []
                    for line in datafile:
                        if len(line.rstrip().split('",',1)[-1][1:-1])>=3:
                            text.append(line.rstrip().split('",',1)[-1][1:-1])
                            idxs.append(line.rstrip().split('",',1)[0][1:-1])
            elif data[-4:]=='.txt':
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
    else:
        raise ValueError('data should be a path or a list of strings')
    
    if preprocessing:
        text = [preprocess(doc) for doc in text]
    
    if return_idxs==True:
        try:
            return text[1:], idxs[1:]
        except NameError:
            print('idxs list not defined')
    else:
        return text[1:]



def _init():
    global init_done
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    init_done = True


def preprocess(text, strip_brackets=False, keep_unicodes=True, add_adj_nn_pairs=True):
    """ 
    * unicodes are extracted before applying stop words and lemmatization
    * !keep?! keep numbers that are attached to words or other textual characters ex: keep 19 in covid19 not in covid 19

    :param str text: the text to be preprocessed
    :param bool strip_brackets: If True, the content inside brackets is excluded
    :param bool keep_unicodes: extract unicodes and aprend them to final output
        NOTE: removes unicodes from unicodes_to_remove. Might wanna extract unocodes from a given list only
    :param bool add_adj_nn_pairs: extract adjective/noun pairs. Append them to result
        NOTE: this implementation keeps adjs and nouns as sole items in final list
              could be implemented with either nltk or spacy
    """
    global init_done

    if not init_done:
        _init()

    if strip_brackets:
        text = re.sub(r'\((.*?)\)', ' ', text)

    if keep_unicodes:
        pattern1 = r'\\U[0-9A-Za-z]{8}'
        pattern2 = r'\\u[0-9A-Za-z]{4}'
        unicodes1 = re.findall(pattern1, text.encode('unicode-escape').decode())
        unicodes2 = re.findall(pattern2, text.encode('unicode-escape').decode())
        unicodes = [u for u in unicodes1 if u not in unicodes_to_remove]
        unicodes.extend([u for u in unicodes2 if u not in unicodes_to_remove])
    
    text = re.sub(r'\b\d+\b', '', text) #or all digits ex: r'\d+'

    text = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text.lower()) # tokenize space seperated group of characters

    lem = nltk.stem.WordNetLemmatizer()

    if add_JJ_NN_pairs_nltk:
        adj_noun_pairs = add_JJ_NN_pairs_nltk(text, lem) #merge adjectives ?

    text = pos_and_lemma(text, lem, adjectives=False)

    text = [w for w in text if w not in nltk.corpus.stopwords.words('english')]

    text = [w for w in text if len(w) >= 3] #ignore with two letters or less

    if add_JJ_NN_pairs_nltk:
        text.extend(adj_noun_pairs)

    if keep_unicodes:
        text.extend(unicodes)
    
    text = ' '.join(text)
    return text


def pos_and_lemma(tokens, lem, adjectives=False):
    # nltk pos tags: https://www.guru99.com/pos-tagging-chunking-nltk.html
    # lem.lemmatize(i,j) -> j: POS tag. Valid options are `"n"` for nouns,`"v"` for verbs, `"a"` for adjectives, `"r"` for adverbs and `"s"` for satellite adjectives.
    result=[]
    for i, j in pos_tag(tokens):
        if j[0].lower() in ['n','v']:
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
            adj_noun_pairs.append(lem.lemmatize(word,'a') + next_word)
    return adj_noun_pairs


def add_JJ_NN_pairs_spacy(text, lem):
    adj_noun_pairs = []
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
            adj_noun_pairs.append(token.text + ' ' + token.head.text)
    return adj_noun_pairs
