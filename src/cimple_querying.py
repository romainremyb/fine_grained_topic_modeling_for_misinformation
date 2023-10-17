from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.plugins.stores.sparqlstore import SPARQLStore
import sys
from collections import defaultdict

from SPARQLWrapper import SPARQLWrapper, JSON


class localGraph:
  # index types by url or class statements

  def __init__(self, folderpath, filenames=['cimple-ontology.ttl', 'cimple-vocabularies.ttl', 'claim-review.ttl',
                                            'mediaeval.ttl', 'propaganda.ttl', 'check-that.ttl', 'birdwatch.ttl',
                                            'afp.ttl']):
    self.g = Graph()
    try:
      for filename in filenames:
        self.g.parse(folderpath+filename, format="turtle")
    except FileNotFoundError as e:
        print(f"File {filename} not found in {folderpath}!", file=sys.stderr)
        return
    
    # add namespace
    cimpler = Namespace("http://data.cimple.eu/")
    self.g.bind("cimple_records", cimpler) 

    self.prefix_namespaces=defaultdict()
    self.namespaces_prefix=defaultdict()
    for prefix, uri in self.g.namespace_manager.namespaces():
        self.prefix_namespaces[prefix]=uri
        self.namespaces_prefix[uri]=prefix
    
    
  def get_full_namespace_dict(self, return_dicts = True):
    self.uris_in_namespaces = defaultdict(lambda: defaultdict(list))
    triples = self.g.triples((None, None, None))
    for s in triples:
      for k in self.namespaces_prefix.keys():
        if str(k) in str(s[0]):
          self.uris_in_namespaces[k]['subjects'].append(s[0])
        if str(k) in str(s[1]):
          self.uris_in_namespaces[k]['predicates'].append(s[1])
        if str(k) in str(s[2]):
          self.uris_in_namespaces[k]['objects'].append(s[2])

    if return_dicts==True:
      return self.uris_in_namespaces
    #TODO save dict with unique uris

  #def records_from_type(self):

  def all_unique_predicates(self): # check if all predicates in prefix_map
    """
    all unique predicates
    """
    triples=self.g.triples((None, None, None))
    all_preds=set()
    for s in triples:
      if s[1] not in all_preds:
        all_preds.add(s[1])
    return all_preds

    
  def get_unique_predicates_for_recordType(self, recordtype, as_subject=True): 
    """
    retrieve unique predicates used with the recordtype as subject or object
    
    : param recordtype:  -> self.prefix_namespaces[prefix] + recordtype (e.g. NewsArticle)
    """
    if as_subject==True:
      triples=self.g.triples((URIRef(recordtype), None, None))
    else:
      triples=self.g.triples((None, None, URIRef(recordtype)))
    predicates=[]
    for s in triples:
      if s[1] not in predicates:
        predicates.append(s[1])
    return predicates
    
    
  def get_all_triples_from_predicate(self, predicate):
    """
    retieve all statements given a predicate
    
    : param predicate:  -> self.prefix_namespaces[prefix] + predicate (e.g. NewsArticle)
    """

    triples = self.g.triples((None, URIRef(predicate), None))
    data = dict()
    for s in triples:
      data[s[0]] = s[2]
    return data


  def get_all_triples_from_predicate_subjectobject(self, predicate, recordType, subject=True):
    """
    retieves all statements given a predicate and a subject/object as recordType
    """
    triples = self.g.triples((None, URIRef(predicate), None))
    data = dict()
    for s in triples: 
      #TODO check subject/object record from type: namespace or statement based -> check all records with namespaces are defined as instances
      data[s[0]]=s[2]
    return data




###

def build_SPARQL_wrapper(endpoint_url):
  sparql = SPARQLWrapper(
      endpoint_url
  )
  sparql.setReturnFormat(JSON)
  return sparql


def request(wrapper, query):
  wrapper.setQuery(query)
  return wrapper.queryAndConvert()["results"]["bindings"]


def get_predicates_recordType(recordtype, as_subject=True): # getsproperties associated with the recordType
  if as_subject==True:
    rt='s'
  else:
    rt='o'
  query = """
SELECT DISTINCT ?predicate
WHERE |
  ?s ?predicate ?o.
  ?{} a schema:{}.
¨
  """.format(rt,recordtype).replace('|','{').replace('¨','}') #TODO: find a better way than using | and ¨
  return query


def get_all_statements_with_predicate(wrapper, predicate):
  query= """
    SELECT ?subject ?object
    WHERE |
    ?subject <{}> ?object.
  ¨
  """.format(predicate)
  return request(wrapper, query)


def get_all_subjects_objects_pairs_from_property_and_subject(wrapper, subject, property):
  query = """
    SELECT ?subject ?object
    WHERE |
    ?subject <{}> ?object .
       ?subject a schema:{} .
    ¨
  """.format(property, subject).replace('|','{').replace('¨','}') #TODO: find a better way than using | and ¨
  return request(wrapper, query)

