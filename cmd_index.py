from preprocessing import create_preprocessor
from indexing import create_index_simple, create_index_spmi

import os

if __name__ == "__main__":
    base_folder = './data/TREC8all/Adhoc/latimes/'
    # document_files = list(map(lambda filename: base_folder + '/' + filename,
    #                       os.listdir(base_folder)))

    document_files = ['./data/TREC8all/Adhoc/latimes/la010289']
    
    print()
    print('Processing {} file(s)'.format(len(document_files)))
    print()

    preprocessor = create_preprocessor(enable_case_folding=True,
                                       enable_remove_stop_words=True,
                                       enable_stemmer=True,
                                       min_length=2)

    print('Writing index using simple method to "simple.index"')
    create_index_simple(document_files, preprocessor, 'simple.index')
    print()
    print('Writing index using SPMI to "spmi.index"')
    create_index_spmi(document_files, preprocessor, 'spmi.index')
