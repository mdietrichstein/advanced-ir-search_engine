from collections import defaultdict, namedtuple
import tempfile

Token = namedtuple('Token', ['term', 'document_frequency', 'document_ids'])


def spmi_invert(token_stream, max_tokens_per_block=1000000):
    """SPMI=Invert implementation

    See https://nlp.stanford.edu/IR-book/html/htmledition/single-pass-in-memory-indexing-1.html
    """

    processed_tokens = 0
    dictionary = defaultdict(list)

    for (doc_id, term) in token_stream:
        postings_list = dictionary[term]

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
        return None, is_exhausted

    # write block to file
    filename = __write_block(dictionary)

    return (filename, is_exhausted)


def merge_blocks(block_filepaths, output_filepath):
    block_files = list(map(lambda filepath: open(filepath, 'r'), block_filepaths))

    head_entries = list(map(lambda file: __read_token(file), block_files))
    head_terms = list(map(lambda entry: entry.term, head_entries))

    num_files = len(block_filepaths)
    num_closed = 0

    with open(output_filepath, 'w') as output_file:
        while num_closed < num_files:

            # Find entries with lexicographical 'smallest' term
            smallest_term = min([t for t in head_terms if t is not None])

            # Get all entries for the given term and merge them
            smallest_idx = [i for i, term in enumerate(head_terms) if term and term == smallest_term]

            merged_document_frequency = 0
            merged_document_ids = []

            for i in smallest_idx:
                token = head_entries[i]

                merged_document_frequency += token.document_frequency
                merged_document_ids.extend(token.document_ids)

                head_entries[i] = __read_token(block_files[i])

                if head_entries[i]:
                    head_terms[i] = head_entries[i].term
                else:
                    block_files[i].close()
                    num_closed += 1

                    head_entries[i] = None
                    head_terms[i] = None

            # Write merged entry to final file
            output_file.write(smallest_term)
            output_file.write('\t')
            output_file.write(str(merged_document_frequency))
            output_file.write('\t')
            output_file.write(','.join(merged_document_ids))
            output_file.write('\n')

    for block_file in block_files:
        if not block_file.closed:
            block_file.close()


def __read_token(file):
    line = file.readline()

    if not line:
        return None  # EOF

    parts = line.split('\t')

    term = parts[0]
    document_frequency = int(parts[1])
    document_ids = parts[2].split(',')

    return Token(term, document_frequency, document_ids)


def __write_block(dictionary):
    """Write the given dictionary to a tmeporary file and returns the filename

    Each dictionary entry is written as a single line, where each line looks
    as follows: '<TERM> <DOCUMENT_FREQUENCY> <DOCUMENT_IDS>'

    * TERM - The term itself
    * DOCUMENT_FREQUENCY - Document Frequency
    * DOCUMENT_IDS - A comma-separated list of documents the term appears in

    The three fields are separated by tabs ('\t')
    """
    default_tmp_dir = tempfile._get_default_tempdir()
    tempfile_name = next(tempfile._get_candidate_names())

    filename = default_tmp_dir + '/' + tempfile_name + '.h5'

    with open(filename, 'w') as f:
        # sort terms
        sorted_terms = sorted(dictionary.keys())

        for term in sorted_terms:
            postings_list = dictionary[term]

            f.write(term)
            f.write('\t')
            f.write(str(len(postings_list)))
            f.write('\t')
            f.write(','.join(postings_list))
            f.write('\n')

    return filename
