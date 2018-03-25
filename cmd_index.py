from preprocessing import create_preprocessor
from indexing import create_index_simple, create_index_spmi

import os

if __name__ == "__main__":
    base_folder = './data/TREC8all/Adhoc/latimes/'
    # document_files = list(map(lambda filename: base_folder + '/' + filename,
    #                       os.listdir(base_folder)))

    document_files = ['./data/TREC8all/Adhoc/latimes/la010289']
    print('Processing {} files'.format(len(document_files)))

    preprocessor = create_preprocessor()
    create_index_simple(document_files, preprocessor, 'simple.index')
    create_index_spmi(document_files, preprocessor, 'spmi.index')
