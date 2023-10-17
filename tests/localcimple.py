import sys, os
from collections import defaultdict
from rdflib import Graph, Literal, URIRef


sys.path.append(os.path.expanduser("~")+'/Desktop/topic_modeling/fine_grained_topic_modeling_for_misinformation/src/')
from cimple_querying import localGraph

cimple = localGraph('KGs_v6/')

uris_in_namespaces = cimple.get_full_namespace_dict()
with open('cimple_metrics.txt','w') as f:
    for k in uris_in_namespaces.keys():
        f.write(cimple.namespaces_prefix[k]+' -> '+str(k)+'\n')
        f.write(f"{len(set(uris_in_namespaces[k]['subjects']))}/{len(uris_in_namespaces[k]['subjects'])}(unique/total) URIs defined as subjects \n")
        f.write(f"{len(set(uris_in_namespaces[k]['predicates']))}/{len(uris_in_namespaces[k]['predicates'])}(unique/total) URIs defined as predicates \n")
        f.write(f"{len(set(uris_in_namespaces[k]['objects']))}/{len(uris_in_namespaces[k]['objects'])}(unique/total) URIs defined as objects \n\n")