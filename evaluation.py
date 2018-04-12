import re
import codecs
import gc
from tqdm import tqdm
from collections import namedtuple

from preprocessing import split_words, create_preprocessor
from searching import simple_tfidf_search, cosine_tfidf_search, simple_bm25_search, simple_bm25va_search


TOP_PATTERN = re.compile(r'<top>(.*?)<\/top>', re.DOTALL | re.M)
START_TAG_PATTERN = re.compile(r'^<(.*?)>.*')

Topic = namedtuple('Topic', ['id', 'title', 'narr', 'desc'])


def generate_qrel(number_of_documents, index, document_stats, topics,
                  output_filepath, ranking_method, run_name):

    print('Generating ranking using', ranking_method)
    topic_scores = []

    for i, topic in enumerate(tqdm(topics)):
        search_terms = topic.title | topic.desc

        document_scores = None

        if ranking_method == 'tfidf':
            document_scores = simple_tfidf_search(number_of_documents, index,
                                                  search_terms)
        elif ranking_method == 'cosine_tfidf':
            document_scores = cosine_tfidf_search(number_of_documents, index,
                                                  search_terms, document_stats)
        elif ranking_method == 'bm25':
            document_scores = simple_bm25_search(number_of_documents, index,
                                                 search_terms, document_stats)
        elif ranking_method == 'bm25va':
            document_scores = simple_bm25va_search(number_of_documents, index,
                                                   search_terms,
                                                   document_stats)

        for document_score in document_scores[:60]:
            topic_scores.append((topic.id, document_score[1], document_score[0]))

        if i % 5 == 0:
            gc.collect()

    with open(output_filepath, 'w') as f:
        for rank, topic_score in enumerate(topic_scores):
            (topic_id, score, document_id) = topic_score

            f.write('{} Q0 {} {} {:6f} {}\n'.format(topic_id, document_id,
                                                    rank+1, score, run_name))


def load_topic_tokens(file_path, encoding='latin-1',
                      strip_html_tags=True,
                      strip_html_entities=True,
                      strip_square_bracket_tags=True,
                      preprocess=create_preprocessor()):
    """Generator which provides a list of topics alogn with the
    preprocessed and tokenized fields
    """
    result = []
    topics = __regex_parse_topics_from_file(file_path)

    for topic in topics:
        processed = {}

        for field in ['title', 'narr', 'desc']:
            content = topic[field]
            words = split_words(content,
                                strip_html_tags=strip_html_tags,
                                strip_html_entities=strip_html_entities,
                                strip_square_bracket_tags=strip_square_bracket_tags)

            terms = preprocess(words)
            processed[field] = set(terms)
       
        result.append(Topic(topic['id'], processed['title'], processed['narr'], processed['desc']))

    return result


def __regex_parse_topics_from_file(file_path, encoding='latin-1'):
    """Loads all topic from the given file using REGEX and returns them
    """

    # read whole file
    content = None
    with codecs.open(file_path, 'r', encoding=encoding) as file:
        content = file.read()

    topics = []

    for topic in TOP_PATTERN.findall(content):
        current_topic = {}
        current_tag_name = None
        current_tag_lines = []

        lines = topic.splitlines()

        for line in lines:
            line = line.strip()

            if line == '':
                continue

            m = re.match(START_TAG_PATTERN, line)

            if m:
                tag = m.group(1)

                if tag == 'top' or tag == '/top':
                    continue

                if tag == 'num':
                    current_topic['id'] = line.replace('<'+tag+'> Number: ', '').strip()
                    continue

                if current_tag_name != None:
                    current_topic[current_tag_name] = ' '.join(current_tag_lines)

                current_tag_name = tag
                current_tag_lines = []

                if current_tag_name == 'title':
                    current_tag_lines.append(line.replace('<'+tag+'> ', '').strip())

                continue

            current_tag_lines.append(line)

        if current_tag_name != None:
            current_topic[current_tag_name] = ' '.join(current_tag_lines)

        topics.append(current_topic)

    return topics
