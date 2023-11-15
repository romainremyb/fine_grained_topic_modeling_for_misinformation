import sys, os
from collections import defaultdict
from rdflib import Graph, Literal, URIRef


sys.path.append(os.path.expanduser("~")+'/Desktop/topic_modeling/fine_grained_topic_modeling_for_misinformation/src/')
from cimple_querying import localGraph

cimple = localGraph('KGs_v6/')

uris_in_namespaces, unique_uris_in_namespaces = cimple.get_full_namespace_dict()

with open('cimple_metrics.txt','w') as f:
    f.write('extracting (subjects/predicates/objects) with http://schema.org/ namespace: \n')

    for place in unique_uris_in_namespaces[URIRef('http://schema.org/')].keys():
        concat_uris="( "
        for uri in unique_uris_in_namespaces[URIRef('http://schema.org/')][place]:
            concat_uris = concat_uris + str(uri) + ' - '
        f.write(f"{place}: \n {concat_uris} ) \n\n")


with open('cimple_metrics.txt','a') as f:
    f.write('\n \nRetrieving record types defined with <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> \n')
    for type in set(cimple.get_all_triples_from_predicate(URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')).values()):
        list_statements = list(cimple.get_unique_uris_from_type_statement(type))
        f.write(f"{type}: there are {len(list_statements)} and {len(set(cimple.get_unique_uris_from_type_namespace(str(list_statements[0]).rsplit('/', 1)[0]+'/')))} unique URIs according to type statements and namespace formatting respectively \n")
    f.write('\n NOTE: ignore: \n http://www.w3.org/2002/07/owl#Ontology -> one unique statements, namespace retrieval looks for http://data.cimple.eu/... \n http://www.w3.org/2004/02/skos/core#Concept retrieved number via namespace is misleading because I striped the last / from records got from type statements...')


with open('cimple_metrics.txt','a') as f:
    f.write('\n\n\n Counts records containing below namespaces in the respective subject/predicate/object position \n')
    for k in uris_in_namespaces.keys():
        f.write(cimple.namespaces_prefix[k]+' -> '+str(k)+'\n')
        f.write(f"{len(set(uris_in_namespaces[k]['subjects']))}/{len(uris_in_namespaces[k]['subjects'])}(unique/total) URIs defined as subjects \n")
        f.write(f"{len(set(uris_in_namespaces[k]['predicates']))}/{len(uris_in_namespaces[k]['predicates'])}(unique/total) URIs defined as predicates \n")
        f.write(f"{len(set(uris_in_namespaces[k]['objects']))}/{len(uris_in_namespaces[k]['objects'])}(unique/total) URIs defined as objects \n\n")


