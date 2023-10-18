from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.plugins.stores.sparqlstore import SPARQLStore
import sys
from collections import defaultdict

from SPARQLWrapper import SPARQLWrapper, JSON


class localGraph:

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
    
    cimpler = Namespace("http://data.cimple.eu/") # add this missing namespace
    self.g.bind("cimple_records", cimpler) 
    tweet = Namespace("http://data.cimple.eu/tweet/") # add this missing namespace
    self.g.bind("tweet", tweet) 
    news = Namespace("http://data.cimple.eu/news-article/") # add this missing namespace
    self.g.bind("news", news) 

    self.prefix_namespaces=defaultdict()
    self.namespaces_prefix=defaultdict()
    for prefix, uri in self.g.namespace_manager.namespaces():
        self.prefix_namespaces[prefix]=uri
        self.namespaces_prefix[uri]=prefix
    
    
  def get_full_namespace_dict(self, return_unique_dict=True, return_dicts = True):
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
    # dictionary with unique URIs
    if return_unique_dict==True:
      self.unique_uris_in_namespaces = defaultdict(lambda: defaultdict(list))
      for pkey in self.uris_in_namespaces.keys():
        for skey in self.uris_in_namespaces[pkey].keys():
          self.unique_uris_in_namespaces[pkey][skey]=list(set(self.uris_in_namespaces[pkey][skey]))

    if return_dicts==True:
      if return_unique_dict==True:
        return self.uris_in_namespaces, self.unique_uris_in_namespaces
      else:
        return self.uris_in_namespaces
    

  def get_unique_uris_from_type_statement(self, type_statement):
    """
    retrieves unique URIs defined as type of type_statement
    
    : param type_statement: URIRef object. ex: self.prefix_namespaces[prefix] + object (e.g. NewsArticle, with namespace: http://schema.org/)
    """
    triples=self.g.triples((None, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), type_statement))
    return set([t[0] for t in triples]) # unique subjects
      

  def get_unique_uris_from_type_namespace(self, uri_namespace, only_as_subjects=False):
    """
    retrieves unique URIs yielding the uri_namespace. 

    : param uri_namespace: URIRef object. namespace of record (e.g. http://data.cimple.eu/news-article/)
    : param only_as_subjects: if True, will only retrieve the records defined as subjects in triples
    """
    triples=self.g.triples((None, None, None))
    uris=[]
    for s in triples:
      if str(uri_namespace) in str(s[0]):
        uris.append(s[0])
      if only_as_subjects==True:
        if str(uri_namespace) in str(s[2]):
          uris.append(s[2])
    return set(uris)


  def all_unique_predicates(self): # check if all predicates in prefix_map
    """
    all unique predicates
    """
    triples=self.g.triples((None, None, None))
    return set([s[1] for s in triples])

    
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
    
    : param predicate:  -> ex: URIRef('http://schema.org/articleBody')
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
    type_uris = self.get_unique_uris_from_type_statement(recordType) #NOTE: could use other function
    data = dict()
    if subject==True:
      for s in triples: 
        if s[0] in type_uris:
          data[s[0]]=s[2]
    else:
      for s in triples: 
        if s[2] in type_uris:
          data[s[0]]=s[2]
    return data







###______________________________________OLD_SPARQL_requests

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

