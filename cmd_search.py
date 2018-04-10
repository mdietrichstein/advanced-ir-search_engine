from preprocessing import create_preprocessor, split_words
from indexing import create_index_reader
from searching import simple_tfidf_search, vecspace_tfidf_search

index_filepath = 'simple.index'

# search_query = 'This is my awesome search term'

search_query = 'Gorbachev policy of glasnost'

preprocess = create_preprocessor(enable_case_folding=True,
                                 enable_remove_stop_words=True,
                                 enable_stemmer=True,
                                 enable_lemmatizer=False,
                                 min_length=2)

words = split_words(search_query,
                    strip_html_tags=True,
                    strip_html_entities=True,
                    strip_square_bracket_tags=True)

search_terms = preprocess(words)

number_of_documents, index_reader_generator = create_index_reader(index_filepath)
index_reader = index_reader_generator()

# read full index
print('reading index')
index = list(index_reader)
number_of_terms = len(index)
print('done')

# document_scores = vecspace_tfidf_search(number_of_documents, index, search_terms)
# for i in range(20):
#   print(document_scores[i], '\n')

document_scores = simple_tfidf_search(number_of_documents, index, search_terms)

for i in range(20):
  print(document_scores[i], '\n')