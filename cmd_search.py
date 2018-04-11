from preprocessing import create_preprocessor, split_words
from indexing import create_index_reader
from searching import simple_tfidf_search, simple_bm25_search, simple_bm25va_search
from evaluation import load_topic_tokens

index_filepath = 'spimi.index'

search_query = 'Gorbachev policy of glasnost'

topics_filepath = './data/TREC8all/topicsTREC8Adhoc.txt'


print('Loading topics from', topics_filepath)
topics = load_topic_tokens(topics_filepath)

print('Loading search index')
number_of_documents, index_reader_generator = create_index_reader(index_filepath)
index_reader = index_reader_generator()
index = list(index_reader)
print('done')

topic = topics[0]
search_terms = topic.title | topic.desc

# document_scores = simple_tfidf_search(number_of_documents, index, search_terms)
document_scores = simple_bm25_search(number_of_documents, index, search_terms)
# document_scores = simple_bm25va_search(number_of_documents, index, search_terms)

for (score, document_id) in document_scores[:50]:
  print('{} {}'.format(score, document_id))