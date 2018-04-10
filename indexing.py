from collections import defaultdict, namedtuple, Counter

import tempfile

from tokenization import generate_tokens_for_files


Token = namedtuple('Token', ['position','term', 'document_frequency', 'postings'])


def create_index_simple(document_files, preprocess, output_filepath,
                        verbose=True,
                        strip_html_tags=True,
                        strip_html_entities=True,
                        strip_square_bracket_tags=True):

    token_stream = generate_tokens_for_files(document_files,
                                             strip_html_tags=strip_html_tags,
                                             strip_html_entities=strip_html_entities,
                                             strip_square_bracket_tags=strip_square_bracket_tags,
                                             preprocess=preprocess)

    token_list = list(token_stream)
    num_documents_processed = token_list[-1][-1]

    # sort by term
    token_list.sort(key=lambda token: token[1])

    current_term = None
    document_ids = []

    with open(output_filepath, 'w') as output_file:
        output_file.write('{}\n'.format(num_documents_processed))

        for (doc_id, term, _) in token_list:
            if term != current_term:
                # we have encountered a new term. write current term to file
                # and reset state
                if document_ids:
                    __write_index_entry(output_file, current_term,
                                        __to_bag_of_words(document_ids))

                current_term = term
                document_ids = []

            document_ids.append(doc_id)

        # write last entry
        if document_ids:
            __write_index_entry(output_file, current_term,
                                __to_bag_of_words(document_ids))


def create_index_spimi(document_files, preprocess, output_filepath,
                       verbose=True,
                       max_tokens_per_block=10000000,
                       strip_html_tags=True,
                       strip_html_entities=True,
                       strip_square_bracket_tags=True):
    """Creates an index using the SPIMI methods
    """

    token_stream = generate_tokens_for_files(document_files,
                                             strip_html_tags=strip_html_tags,
                                             strip_html_entities=strip_html_entities,
                                             strip_square_bracket_tags=strip_square_bracket_tags,
                                             preprocess=preprocess)

    block_filenames = []
    is_exhausted = False

    num_documents_processed = 0

    while not is_exhausted:
        filename, is_exhausted, num_documents_processed = \
            __spimi_invert(token_stream, max_tokens_per_block=max_tokens_per_block)

        block_filenames.append(filename)

    if verbose:
        print('Merging {} block(s)'.format(len(block_filenames)))

    with open(output_filepath, 'w') as output_file:
        output_file.write('{}\n'.format(num_documents_processed))
        __merge_spimi_blocks(output_file, block_filenames)


def __spimi_invert(token_stream, max_tokens_per_block):
    """SPIMI-Invert implementation

    See https://nlp.stanford.edu/IR-book/html/htmledition/single-pass-in-memory-indexing-1.html
    """

    processed_tokens = 0
    dictionary = defaultdict(list)

    for (doc_id, term, num_documents_processed) in token_stream:
        #  returns empty list if term is not yet present (defaultdict)
        postings_list = dictionary[term]
        postings_list.append(doc_id)

        processed_tokens += 1

        if processed_tokens >= max_tokens_per_block:
            break

    # we have reached the end of the token_stream
    is_exhausted = processed_tokens < max_tokens_per_block

    # return empty filename if block is empty
    if not dictionary:
        return (None, is_exhausted, num_documents_processed)

    # write block to file
    filename = __write_spimi_block(dictionary)

    return (filename, is_exhausted, num_documents_processed)


def __merge_spimi_blocks(output_file, block_filepaths):
    block_files = list(map(lambda filepath: open(filepath, 'r'), block_filepaths))

    head_entries = list(map(lambda file: __read_token(file), block_files))
    head_terms = list(map(lambda entry: entry.term, head_entries))

    num_files = len(block_filepaths)
    num_closed = 0

    while num_closed < num_files:

        # Find entries with lexicographical 'smallest' term
        smallest_term = min([t for t in head_terms if t is not None])

        # Get all entries for the given term and merge them
        smallest_idx = [i for i, term in enumerate(head_terms) if term and term == smallest_term]

        merged_postings = Counter()

        for i in smallest_idx:
            token = head_entries[i]

            merged_postings += Counter(dict(token.postings))

            head_entries[i] = __read_token(block_files[i])

            if head_entries[i]:
                head_terms[i] = head_entries[i].term
            else:
                block_files[i].close()
                num_closed += 1

                head_entries[i] = None
                head_terms[i] = None

        __write_index_entry(output_file, smallest_term, merged_postings.items())

    for block_file in block_files:
        if not block_file.closed:
            block_file.close()


def create_index_reader(filepath):
    """ Returns index stats (number of documents) and a generator for iterating 
    over each index entry
    """
    f = open(filepath)
    
    number_of_documents = int(f.readline())

    def generator():
        position = 0
        with f:
            token = __read_token(f, position)

            while token:
                yield token
                position += 1
                token = __read_token(f, position)

    return (number_of_documents, generator)


def __read_token(file, position=None):
    line = file.readline()

    if not line:
        return None  # EOF

    parts = line.split('\t')

    term = parts[0]
    document_frequency = int(parts[1])

    postings_entries = parts[2].split(',')

    def to_tuple(e):
        p = e.split('|')
        return (p[0], int(p[1]))  # (document_id, term_freq)

    postings = list(map(to_tuple, postings_entries))

    return Token(position, term, document_frequency, postings)


def __write_spimi_block(dictionary):
    """Write the given dictionary to a temporary file and returns the filename

    See __write_index_entry for details on how an entry is serialized
    """
    default_tmp_dir = tempfile._get_default_tempdir()
    tempfile_name = next(tempfile._get_candidate_names())

    filename = default_tmp_dir + '/' + tempfile_name + '.blk'

    with open(filename, 'w') as f:
        # sort terms
        sorted_terms = sorted(dictionary.keys())

        for term in sorted_terms:
            __write_index_entry(f, term, __to_bag_of_words(dictionary[term]))

    return filename


def __write_index_entry(file, term, postings_list):
    """Writes s single index entry into the given file

    An entry looks as follows: '<TERM> <DOCUMENT_FREQUENCY> <POSTINGS>'

    * TERM - The term itself
    * DOCUMENT_FREQUENCY - Document Frequency (Number of documents the term appears in)
    * POSTINGS - A comma-separated list of documents the term appears in
      along with the term frequency separated by pipe in the given document:
      <DOCUMENT_ID>|<TERM_FREQUENCY>,<DOCUMENT_ID>|<TERM_FREQUENCY>,...
    * TERM_FREQUENCY - Number of times the term appears in the corresponding document 

    """
    postings = list(map(lambda e: '{}|{}'.format(e[0], e[1]), postings_list))

    line = '{}\t{}\t{}\n'.format(
        term, str(len(postings_list)), ','.join(postings))

    file.write(line)


def __to_bag_of_words(words):
    return Counter(words).items()
