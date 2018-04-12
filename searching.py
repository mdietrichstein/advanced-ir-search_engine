import math
from collections import namedtuple, Counter

Document = namedtuple('Document', ['id', 'terms'])


def simple_tfidf_search(number_of_documents, index, search_terms):
    """Runs a simple tf-idf search through the index

    Optimization: Pre-calculate tf / idf score and store it in index
    """
    search_terms = list(set(search_terms))

    tokens = __find_tokens_for_terms(index, search_terms)
    document_scores = Counter()

    for token in tokens:
        for document_id, tfd in token.postings:
            document_scores[document_id] += __tfidf_score(number_of_documents,
                                                          token.document_frequency,
                                                          tfd)

    document_scores = list(document_scores.items())
    document_scores.sort(key=lambda ds: ds[1], reverse=True)

    return document_scores


def cosine_tfidf_search(number_of_documents, index, search_terms):
    """Runs a cosine tf-idf search through the index
    """
    search_term_counter = Counter(search_terms)

    tokens = __find_tokens_for_terms(index, search_terms)

    document_scores = Counter()

    document_norms = Counter()
    query_norm = 0

    for token in tokens:
        tfq = search_term_counter[token.term]

        w_tq = __tfidf_score(number_of_documents, token.document_frequency,
                             tfq)
        query_norm += w_tq * w_tq

        for document_id, tfd in token.postings:
            w_tf = __tfidf_score(number_of_documents, token.document_frequency,
                                 tfd)
            document_scores[document_id] += w_tq * w_tf
            document_norms[document_id] += w_tf * w_tf

    query_norm = math.sqrt(query_norm)

    for document_id in document_scores:
        document_norm = math.sqrt(document_norms[document_id])
        document_scores[document_id] /= (document_norm * query_norm)

    document_scores = list(document_scores.items())
    document_scores.sort(key=lambda ds: ds[1], reverse=True)
    return document_scores


def simple_bm25_search(number_of_documents, index, search_terms,
                       document_stats, k1=1.2, b=0.75, k3=100):
    """Runs a simple bm25 search through the index
    """
    tokens = __find_tokens_for_terms(index, search_terms)

    document_scores = Counter()
    document_length_counter = document_stats['length']

    average_document_length = __calculate_average_document_length(document_length_counter)

    search_term_counter = Counter(search_terms)

    for token in tokens:
        tfq = search_term_counter[token.term]
        dft = token.document_frequency

        for (document_id, tfd) in token.postings:
            document_length = document_length_counter[document_id]

            length_ratio = (document_length / average_document_length)
            Bd = ((1 - b) + (b * length_ratio))

            document_scores[document_id] += __bm25_score(number_of_documents,
                                                         tfq, tfd, dft,
                                                         Bd, k1, k3)

    document_scores = list(document_scores.items())
    document_scores.sort(key=lambda ds: ds[1], reverse=True)
    return document_scores


def simple_bm25va_search(number_of_documents, index, search_terms,
                         document_stats, k1=1.2, k3=100):
    """Runs a simple bm25va search through the index
    """
    tokens = __find_tokens_for_terms(index, search_terms)

    document_scores = Counter()
    document_terms_counter = document_stats['terms']
    document_length_counter = document_stats['length']

    average_document_length = __calculate_average_document_length(document_length_counter)
    mean_average_term_frequency = __calculate_mean_average_term_frequency(document_length_counter, document_terms_counter)

    search_term_counter = Counter(search_terms)

    for token in tokens:
        tfq = search_term_counter[token.term]
        dft = token.document_frequency

        for (document_id, tfd) in token.postings:
            document_length = document_length_counter[document_id]

            length_ratio = (document_length / average_document_length)
            Bva =  1 / (mean_average_term_frequency * mean_average_term_frequency)
            Bva *= (document_length / document_terms_counter[document_id])
            Bva += (1 - (1 / mean_average_term_frequency)) * length_ratio

            document_scores[document_id] += __bm25_score(number_of_documents,
                                                         tfq, tfd, dft,
                                                         Bva, k1, k3)

    document_scores = list(document_scores.items())
    document_scores.sort(key=lambda ds: ds[1], reverse=True)
    return document_scores


def __calculate_mean_average_term_frequency(document_length_counter,
                                            document_terms_counter):
    document_ids = list(document_length_counter.keys())
    return sum([__calculate_average_term_frequency(document_id,
                                                   document_length_counter,
                                                   document_terms_counter) for document_id in document_ids]) / len(document_ids)


def __calculate_average_term_frequency(document_id, document_length_counter,
                                       document_terms_counter):
    return document_length_counter[document_id] / document_terms_counter[document_id]


def __calculate_average_document_length(document_length_counter):
    lengths = list(document_length_counter.values())
    return sum(lengths) / len(lengths)


def __tfidf_score(number_of_documents, document_frequency, term_frequency):
    idf = math.log(number_of_documents / document_frequency)
    return math.log(1 + term_frequency) * idf


def __bm25_score(number_of_documents, tfq, tfd, dft, B, k1, k3):
    K = k1 * B

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
