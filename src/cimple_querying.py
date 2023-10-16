from rdflib import Graph
from rdflib.plugins.stores.sparqlstore import SPARQLStore

from SPARQLWrapper import SPARQLWrapper, JSON



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
    WHERE {
    ?subject <{}> ?object .
       ?subject a schema:{} .
    ¨
  """.format(property, subject).replace('|','{').replace('¨','}') #TODO: find a better way than using | and ¨
  return request(wrapper, query)

