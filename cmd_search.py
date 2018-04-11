from preprocessing import create_preprocessor, split_words
from evaluation import generate_qrel

index_filepath = 'spimi.index'
topics_filepath = './data/TREC8all/topicsTREC8Adhoc.txt'

ranking_method = 'tfidf'
generate_qrel(index_filepath, topics_filepath,
              f'{ranking_method}_results.txt',
              ranking_method, 'dev-run')

ranking_method = 'bm25'
generate_qrel(index_filepath, topics_filepath,
              f'{ranking_method}_results.txt',
              ranking_method, 'dev-run')