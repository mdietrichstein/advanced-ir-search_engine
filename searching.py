import math
from collections import namedtuple, Counter

Document = namedtuple('Document', ['id', 'terms'])


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


def cosine_tfidf_search(number_of_documents, index, search_terms,
                        document_stats):
    """Runs a cosine tf-idf search through the index
    """
    search_term_counter = Counter(search_terms)

    tokens = __find_tokens_for_terms(index, search_terms)

    document_scores = Counter()

    for token in tokens:
        tfq = search_term_counter[token.term]

        w_tq = __tfidf_score(number_of_documents, token.document_frequency,
                             tfq)

        for document_id, tfd in token.postings:
            w_tf = __tfidf_score(number_of_documents, token.document_frequency,
                                 tfd)
            document_scores[document_id] += w_tf * w_tq

    document_length_counter = document_stats['length']

    for document_id in document_scores:
        normalized_score = document_scores[document_id] / document_length_counter[document_id]
        document_scores[document_id] = normalized_score

    document_scores = list(document_scores.items())
    document_scores.sort(key=lambda ds: ds[1], reverse=True)

    return document_scores


def simple_bm25_search(number_of_documents, index, search_terms,
                       document_stats, k1=1.2, b=0.75, k3=8):
    """Runs a simple bm25 search through the index
    """
    (documents, tokens) = __find_documents_for_terms(index, search_terms)
    document_scores = []

    document_list = documents.values()

    document_length_counter = document_stats['length']

    # Optimization: Save average_document_length in index
    average_document_length = __calculate_average_document_length(document_list, document_length_counter)

    search_term_counter = Counter(search_terms)

    for document in document_list:
        # Optimization: Save document length in index
        document_length = document_length_counter[document.id]

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

            score += __bm25_score(number_of_documents,
                                  tfq, tfd, dft,
                                  Bd, k1, k3)

        document_scores.append([document.id, score])

    document_scores.sort(key=lambda ds: ds[1], reverse=True)

    return document_scores


def simple_bm25va_search(number_of_documents, index, search_terms,
                         document_stats, k1=1.2, k3=8):
    """Runs a simple bm25va search through the index
    """
    (documents, tokens) = __find_documents_for_terms(index, search_terms)
    document_scores = []

    document_list = documents.values()

    document_terms_counter = document_stats['terms']
    document_length_counter = document_stats['length']

    # Optimization: Save average_document_length in index
    average_document_length = __calculate_average_document_length(document_list, document_length_counter)
    mean_average_term_frequency = __calculate_mean_average_term_frequency(document_list, document_length_counter, document_terms_counter)

    search_term_counter = Counter(search_terms)

    for document in document_list:
        # Optimization: Save document length in index
        document_length = document_length_counter[document.id]

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
            Bva += (1 - (1 / mean_average_term_frequency)) * length_ratio

            score += __bm25_score(number_of_documents,
                                  tfq, tfd, dft,
                                  Bva, k1, k3)

        document_scores.append([document.id, score])

    document_scores.sort(key=lambda ds: ds[1], reverse=True)

    return document_scores


def __calculate_mean_average_term_frequency(documents, document_length_counter, document_terms_counter):
    return sum([__calculate_average_term_frequency(document, document_length_counter, document_terms_counter) for document in documents]) / len(documents)


def __calculate_average_term_frequency(document, document_length_counter,
                                       document_terms_counter):
    return document_length_counter[document.id] / document_terms_counter[document.id]


def __calculate_average_document_length(documents, document_length_counter):
    avg = 0

    for document in documents:
        avg += document_length_counter[document.id]

    return avg / len(documents)


def __tfidf_score(number_of_documents, document_frequency, term_frequency):
    idf = math.log(number_of_documents / document_frequency)
    return math.log(1 + term_frequency) * idf


def __bm25_score(number_of_documents, tfq, tfd, dft, Bd, k1, k3):
    K = k1 * Bd

    bm25 = ((k3+1)*tfq)/(k3+tfq)
    bm25 *= (((k1+1)*tfd)/(K+tfd))
    bm25 *= math.log((number_of_documents-dft+0.5)/(dft+0.5))

    return bm25


def __find_tokens_for_terms(index, search_terms):
    """Returns matching token objects for the given terms
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

    return search_tokens


def __find_documents_for_terms(index, search_terms):
    """Returns matching documents and token objects for
    the given terms
    """
    search_tokens = __find_tokens_for_terms(index, search_terms)

    documents = {}

    # create document -> term mapping
    for token in search_tokens:
        for posting in token.postings:

            document_id = posting[0]
            term_frequency = posting[1]

            if not document_id in documents:
                documents[document_id] = Document(document_id, {})

            document = documents[document_id]
            document.terms[token.term] = term_frequency

    return (documents, search_tokens)
