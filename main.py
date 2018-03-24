from tokenization import generate_tokens_for_files
from indexing import spmi_invert, merge_blocks

import os

if __name__ == "__main__":
    base_folder = './data/TREC8all/Adhoc/latimes/'
    document_files = list(map(lambda filename: base_folder + '/' + filename,
                          os.listdir(base_folder)))

    # document_files = ['./data/TREC8all/Adhoc/latimes/la010189']
    token_stream = generate_tokens_for_files(document_files)

    print('Processing {} files'.format(len(document_files)))

    block_filenames = []
    is_exhausted = False

    while not is_exhausted:
        filename, is_exhausted = spmi_invert(token_stream,
                                             max_tokens_per_block=10000000)

        block_filenames.append(filename)

    print('Merging {} blocks'.format(len(block_filenames)))
    merge_blocks(block_filenames, 'merged.index')
