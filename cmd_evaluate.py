from preprocessing import create_preprocessor, split_words
from evaluation import generate_qrel, load_topic_tokens
from indexing import create_index_reader, load_document_stats
import gc

index_filepath = 'spimi.index'
stats_filepath = 'spimi.stats'
topics_filepath = './data/TREC8all/topicsTREC8Adhoc.txt'

preprocessor = create_preprocessor(enable_case_folding=True,
                                   enable_remove_stop_words=True,
                                   enable_stemmer=True,
                                   enable_lemmatizer=False,
                                   min_length=2)

print('Loading topics from', topics_filepath)
topics = load_topic_tokens(topics_filepath, preprocess=preprocessor)
print('Searching', len(topics), 'topics')

print('Loading document stats')
document_stats = load_document_stats(stats_filepath)
print('done')

print('Loading search index')
number_of_documents, index_reader_generator = create_index_reader(index_filepath)
index_reader = index_reader_generator()
index = list(index_reader)
print('done')

ranking_method = 'tfidf'
generate_qrel(number_of_documents, index, document_stats, topics,
              f'{ranking_method}_results.txt',
              ranking_method, 'dev-run')
gc.collect()

ranking_method = 'cosine_tfidf'
generate_qrel(number_of_documents, index, document_stats, topics,
              f'{ranking_method}_results.txt',
              ranking_method, 'dev-run')
gc.collect()

ranking_method = 'bm25'
generate_qrel(number_of_documents, index, document_stats, topics,
              f'{ranking_method}_results.txt',
              ranking_method, 'dev-run')
gc.collect()

ranking_method = 'bm25va'
generate_qrel(number_of_documents, index, document_stats, topics,
              f'{ranking_method}_results.txt',
              ranking_method, 'dev-run')
gc.collect()
