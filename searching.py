import math
from collections import namedtuple, Counter

Document = namedtuple('Document', ['id', 'terms', 'length', 'number_of_terms'])

# def vecspace_tfidf_search(number_of_documents, index, search_terms):
#     search_terms = list(set(search_terms))
#     (documents, search_tokens) = __find_documents_for_terms(index, search_terms)
#     document_scores = []

#     document_list = documents.values()

#     number_of_search_terms = len(search_terms)

#     document_vectors = np.zeros((number_of_documents, number_of_search_terms))

#     for di, document in enumerate(document_list):
#       document_vector = document_vectors[di]

#       for ti, search_token in enumerate(search_tokens):

#         if search_token.term in document.terms:
#           term_frequency = document.terms[search_token.term]

#         document_vector[ti] = \
#           __tfidf_score(number_of_documents,
#                         search_token.document_frequency,
#                         term_frequency)

#         norm = np.linalg.norm(document_vector)
#         document_vectors[di] = document_vector / norm

#     query_vector = np.zeros((number_of_search_terms, 1))

#     for ti, search_token in enumerate(search_tokens):
#       query_vector[ti] = math.log(number_of_documents / search_token.document_frequency)


#     scores = np.dot(document_vectors, query_vector)

#     idx = (-scores).argsort()[:50]
#     return np.array(document_list)[idx]


def simple_tfidf_search(number_of_documents, index, search_terms):
    """Runs a simple tf-idf search through the index

    Optimization: Pre-calculate tf-idf score and store it in index
    """
    search_terms = list(set(search_terms))

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

            score += __tfidf_score(number_of_documents, document_frequency,
                                   term_frequency)

        document_scores.append([document.id, score])

    document_scores.sort(key=lambda ds: ds[1], reverse=True)

    return document_scores


def simple_bm25_search(number_of_documents, index, search_terms,
                       k1=1.2, b=0.75, k3=100):
    """Runs a simple bm25 search through the index
    """
    (documents, tokens) = __find_documents_for_terms(index, search_terms, include_stats=True)
    document_scores = []

    document_list = documents.values()

    # Optimization: Save average_document_length in index
    average_document_length = __calculate_average_document_length(document_list)

    search_term_counter = Counter(search_terms)

    for document in document_list:
        # Optimization: Save document length in index
        document_length = __document_length(document)

        score = 0

        for token in tokens:
            term_frequency = 0

            if token.term in document.terms:
                term_frequency = document.terms[token.term]

            tfq = search_term_counter[token.term]
            tfd = term_frequency
            dft = token.document_frequency

            length_ratio = (document_length / average_document_length)
            Bd = ((1 - b) + (b * length_ratio))
            K = k1 * Bd

            bm25 = ((k3+1)*tfq)/(k3+1)
            bm25 *= (((k1+1)*tfd)/(K+tfd))
            bm25 *= math.log((number_of_documents-dft+0.5)/(dft+0.5))

            score += bm25

        document_scores.append([document.id, score])

    document_scores.sort(key=lambda ds: ds[1], reverse=True)

    return document_scores


def simple_bm25va_search(number_of_documents, index, search_terms,
                         k1=1.2, k3=100):
    """Runs a simple bm25va search through the index
    """
    (documents, tokens) = __find_documents_for_terms(index, search_terms, include_stats=True)
    document_scores = []

    document_list = documents.values()

    # Optimization: Save average_document_length in index
    average_document_length = __calculate_average_document_length(document_list)
    mean_average_term_frequency = __calculate_mean_average_term_frequency(document_list)

    search_term_counter = Counter(search_terms)

    for document in document_list:
        # Optimization: Save document length in index
        document_length = __document_length(document)

        score = 0

        for token in tokens:
            term_frequency = 0

            if token.term in document.terms:
                term_frequency = document.terms[token.term]

            tfq = search_term_counter[token.term]
            tfd = term_frequency
            dft = token.document_frequency

            length_ratio = (document_length / average_document_length)
            Bva =  1 / (mean_average_term_frequency * mean_average_term_frequency)
            Bva *= (document_length / len(document.terms))
            Bva += 1 - (1 / mean_average_term_frequency)
            Bva *= length_ratio

            K = k1 * Bva

            bm25va = ((k3+1)*tfq)/(k3+1)
            bm25va *= (((k1+1)*tfd)/(K+tfd))
            bm25va *= math.log((number_of_documents-dft+0.5)/(dft+0.5))

            score += bm25va

        document_scores.append([document.id, score])

    document_scores.sort(key=lambda ds: ds[1], reverse=True)

    return document_scores


def __calculate_mean_average_term_frequency(documents):
    return sum([__calculate_average_term_frequency(document) for document in documents]) / len(documents)


def __calculate_average_term_frequency(document):
    return document.length / document.number_of_terms


def __calculate_average_document_length(documents):
    avg = 0

    for document in documents:
        avg += __document_length(document)

    return avg / len(documents)


def __document_length(document):
    return document.length


def __tfidf_score(number_of_documents, document_frequency, term_frequency):
    idf = math.log(number_of_documents / document_frequency)
    return term_frequency * idf


def __find_documents_for_terms(index, search_terms, include_stats=False):
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
    for token in search_tokens:
        for posting in token.postings:

            document_id = posting[0]
            term_frequency = posting[1]

            if not document_id in documents:
                documents[document_id] = Document(document_id, {}, 0, 0)

            document = documents[document_id]
            document.terms[token.term] = term_frequency

    if include_stats:
        for i, token in enumerate(index):
            if token.term in needles:
                continue

            for document_id, term_frequency in token.postings:
                if document_id in documents:
                    document = documents[document_id]

                    document_length = document.length
                    number_of_terms = document.number_of_terms

                    if not document_length:
                        document_length = sum(list(document.terms.values()))
                        number_of_terms = len(document.terms)

                    document_length += term_frequency
                    number_of_terms += 1

                    document = Document(document_id, document.terms,
                                        document_length,
                                        number_of_terms)

                    documents[document_id] = document

    return (documents, search_tokens)
