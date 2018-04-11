from preprocessing import create_preprocessor
from indexing import create_index_simple, create_index_spimi

import os
import glob
import nltk

if __name__ == "__main__":
    nltk.download('wordnet')

    base_folder = './data/TREC8all/Adhoc/'
    glob_pattern = base_folder + '/**'

    document_files = [fname for fname in glob.glob(glob_pattern, recursive=True) if os.path.isfile(fname)]

    print()
    print('Processing {} file(s)'.format(len(document_files)))
    print()

    preprocessor = create_preprocessor(enable_case_folding=True,
                                       enable_remove_stop_words=True,
                                       enable_stemmer=True,
                                       enable_lemmatizer=False,
                                       min_length=2)

    # print('Writing index using simple method to "simple.index"')
    # print('Reading source files')
    # create_index_simple(document_files, preprocessor, 'simple.index')
    # print()
    print('Writing index using SPIMI to "spimi.index"')
    print('Reading source files')
    create_index_spimi(document_files, preprocessor, 'spimi.index', max_tokens_per_block=10000000)
