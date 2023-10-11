from rdflib import Graph
from rdflib.plugins.stores.sparqlstore import SPARQLStore


def build_SPARQL_wrapper(endpoint_url):
  store = SPARQLStore(endpoint=endpoint_url)
  g = Graph(store)
  return g

def request(wrapper, query):
  return wrapper.query(query)


def get_predicates_ofrecordType_query(recordtype): # getsproperties associated with the recordType
  query = """
    SELECT DISTINCT ?predicate
    WHERE |
      ?s a schema:{};
        ?predicate ?o.
    ¨
  """.format(recordtype).replace('|','{').replace('¨','}') #TODO: find a better way than using | and ¨
  return query


def get_all_subjects_objects_pairs_from_property_and_subject(wrapper, property, subject):
  query = """
    SELECT ?s ?object
    WHERE |
      ?s a schema:{};
        schema:{} ?object.
    ¨
  """.format(subject, property).replace('|','{').replace('¨','}') #TODO: find a better way than using | and ¨
  return request(wrapper, query)