import sys, os
import json
from collections import defaultdict
from rdflib import Graph, Literal, URIRef
import csv
from cimple_querying import localGraph


cimple = localGraph('KGs_v6/')

uris_in_namespaces, unique_uris_in_namespaces = cimple.get_full_namespace_dict()

os.chdir(os.path.expanduser("~")+'/Desktop/topic_modeling/fine_grained_topic_modeling_for_misinformation/ToMODAPI_data/')

type_statements = cimple.get_all_triples_from_predicate(URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'))
predicates_per_types = defaultdict(dict)
for i in set(type_statements.values()):
    predicates_per_types[i]['as_subjects']=cimple.get_unique_predicates_for_recordType(i, as_subject=True)
    predicates_per_types[i]['as_objects']=cimple.get_unique_predicates_for_recordType(i, as_subject=False)


twitts = cimple.get_all_triples_from_predicate_subjectobject(URIRef('http://schema.org/text'), URIRef('http://schema.org/SocialMediaPosting'), subject=True)
twitter_data = dict()
twitter_data['id_doc']=[]
twitter_data['content']=[]
for uri in twitts.keys():
    twitter_data['id_doc'].append(str(uri).rsplit('/', 1)[-1])
    twitter_data['content'].append(str(twitts[uri]))

with open(str(os.getcwd()).rsplit('/', 1)[0]+'/data/twitts.csv', 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL, quotechar='"')
    writer.writerow(twitter_data.keys())
    writer.writerows(zip(*twitter_data.values()))

afp = cimple.get_all_triples_from_predicate_subjectobject(URIRef('http://schema.org/articleBody'), URIRef('http://schema.org/NewsArticle'), subject=True)
afp_data = dict()
afp_data['id_doc']=[]
afp_data['content']=[]
for uri in afp.keys():
    afp_data['id_doc'].append(str(uri).rsplit('/', 1)[-1])
    afp_data['content'].append(str(afp[uri]))

with open(str(os.getcwd()).rsplit('/', 1)[0]+'/data/afp.csv', 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL, quotechar='"')
    writer.writerow(afp_data.keys())
    writer.writerows(zip(*afp_data.values()))