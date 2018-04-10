import math
import numpy as np
from collections import namedtuple

Document = namedtuple('Document', ['id', 'terms'])

def simple_tfidf_search(number_of_documents, index, search_terms):
  """Runs a simple tf-idf search thourgh the index
  """
  (documents, tokens) = __find_documents_for_terms(index, search_terms)
  document_scores = []

  document_list = documents.values()

  # calculate tf-idf scores
  for document in document_list:
    score = 0

    for token in tokens:
      term_frequency = 0

      if token.term in document.terms:
        term_frequency = document.terms[token.term]

      document_frequency = token.document_frequency

      score += __tfidf_score(number_of_documents, document_frequency, term_frequency)

    document_scores.append([document, score])

  document_scores.sort(key=lambda ds: ds[1], reverse=True)

  return document_scores


def __tfidf_score(number_of_documents, document_frequency, term_frequency):
  idf = math.log(number_of_documents / document_frequency)
  return term_frequency * idf


def __find_documents_for_terms(index, search_terms):
  """Returns matching documents and token objects for
  the given terms
  """
  search_tokens = []
  needles = set(search_terms)

  for token in index:
    for needle in needles:
      if needle == token.term:
        search_tokens.append(token)

        remaining_needles = list(needles)
        remaining_needles.remove(needle)
        needles = remaining_needles

      if not needles:
        break

  documents = {}

  # create document -> term mapping
  for search_token in search_tokens:
    for posting in search_token.postings:

      document_id = posting[0]
      term_frequency = posting[1]

      if not document_id in documents:
        documents[document_id] = Document(document_id, {})

      document = documents[document_id]
      document.terms[search_token.term] = term_frequency

  return (documents, search_tokens)
