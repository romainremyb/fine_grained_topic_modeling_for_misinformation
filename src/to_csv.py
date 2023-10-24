import sys, os
import json
from collections import defaultdict
from rdflib import Graph, Literal, URIRef
import csv
from cimple_querying import localGraph


cimple = localGraph('KGs_v6/')

uris_in_namespaces, unique_uris_in_namespaces = cimple.get_full_namespace_dict()

os.chdir(os.path.expanduser("~")+'/Desktop/topic_modeling/fine_grained_topic_modeling_for_misinformation/data/')

type_statements = cimple.get_all_triples_from_predicate(URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'))
predicates_per_types = defaultdict(dict)
for i in set(type_statements.values()):
    predicates_per_types[i]['as_subjects']=cimple.get_unique_predicates_for_recordType(i, as_subject=True)
    predicates_per_types[i]['as_objects']=cimple.get_unique_predicates_for_recordType(i, as_subject=False)


data = dict()
data['id_doc']=[]
data['content']=[]
for pair in [(URIRef('http://schema.org/text'), URIRef('http://schema.org/SocialMediaPosting')), 
              (URIRef('http://schema.org/text'), URIRef('http://schema.org/Claim'))]:
    statements = cimple.get_all_triples_from_predicate_subjectobject(pair[0], pair[1], subject=True)
    for uri in statements.keys():
        data['id_doc'].append(str(uri).rsplit('/', 1)[-1])
        data['content'].append(str(statements[uri]))


with open(str(os.getcwd()).rsplit('/', 1)[0]+'/data/data.csv', 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL, quotechar='"')
    writer.writerow(data.keys())
    writer.writerows(zip(*data.values()))

