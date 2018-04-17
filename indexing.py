import json
import tempfile
import gc
import os
import glob
import shutil
from collections import defaultdict, namedtuple, Counter
from pathos.multiprocessing import ProcessingPool
from tokenization import generate_tokens_for_files, generate_tokens_for_files_distributed


Token = namedtuple('Token', ['position', 'term', 'document_frequency', 'postings'])


def create_index_simple(document_files, preprocess, output_filepath,
                        document_stats_path,
                        verbose=True,
                        strip_html_tags=True,
                        strip_html_entities=True,
                        strip_square_bracket_tags=True):

    token_stream = generate_tokens_for_files(document_files,
                                             strip_html_tags=strip_html_tags,
                                             strip_html_entities=strip_html_entities,
                                             strip_square_bracket_tags=strip_square_bracket_tags,
                                             preprocess=preprocess)

    document_terms_counter = Counter()
    document_length_counter = Counter()

    token_list = list(token_stream)
    num_documents_processed = token_list[-1][-1]

    # sort by term
    token_list.sort(key=lambda token: token[1])

    current_term = None
    document_ids = []

    with open(output_filepath, 'w') as output_file:
        output_file.write('{}\n'.format(num_documents_processed))

        for i, (doc_id, term, _) in enumerate(token_list):
            if term != current_term:
                # we have encountered a new term. write current term to file
                # and reset state
                if document_ids:
                    __flush_index_entry(output_file, current_term,
                                        __to_bag_of_words(document_ids),
                                        document_terms_counter,
                                        document_length_counter)

                current_term = term
                document_ids = []

            document_ids.append(doc_id)

            if i % 50000 == 0:
                gc.collect()

        # write last entry
        if document_ids:
            __flush_index_entry(output_file, current_term,
                                __to_bag_of_words(document_ids),
                                document_terms_counter,
                                document_length_counter)

    __write_document_stats(document_stats_path,
                           document_terms_counter,
                           document_length_counter)


def create_index_spimi(document_files, preprocess, output_filepath,
                       document_stats_path,
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
        print('This might take a while...')

    with open(output_filepath, 'w') as output_file:
        output_file.write('{}\n'.format(num_documents_processed))
        __merge_spimi_blocks(output_file, document_stats_path, block_filenames)


def create_index_map_reduce(document_files, preprocess, output_filepath,
                        document_stats_path,
                        verbose=True,
                        strip_html_tags=True,
                        strip_html_entities=True,
                        strip_square_bracket_tags=True,
                        blocksize=16,
                        num_nodes=None):

    def __setup():
        if os.path.isdir(segment_path):
            for x in glob.glob(segment_path+"*"):
                os.remove(x)
        else:
            os.mkdir(segment_path)

        if os.path.isdir(posting_path):
            for x in glob.glob(posting_path+"*"):
                os.remove(x)
        else:
            os.mkdir(posting_path)

    def __down():
        for x in glob.glob(segment_path+"*"):
            os.remove(x)
        for x in glob.glob(posting_path+"*"):
            os.remove(x)
        os.rmdir(segment_path)
        os.rmdir(posting_path)
        pass

    posting_path = "./postings/"
    segment_path = "./segmented_files/"

    splitsize = 1048576 * blocksize
    splits = []
    split = []
    current_size = 0

    partitions = ["aa", "bc", "de", "fh", "ij", "km", "nq", "rs", "tu", "vz"]

    if verbose:
        print("Setting up directories...")

    __setup()

    if verbose:
        print("Splitting up tasks...")

    for f in document_files:
        if os.path.isfile(f):
            current_size += os.path.getsize(f)
            if current_size <= splitsize:
                split.append(f)
            else:
                splits.append(list(split))
                current_size = os.path.getsize(f)
                split.clear()
                split.append(f)

    pool = ProcessingPool(nodes=num_nodes)
    mul = splits.__len__()

    if verbose:
        print("Starting Map Phase...".format(num_nodes))

    pool.map(__map, splits, [strip_html_tags]*mul,
             [strip_html_entities]*mul,
             [strip_square_bracket_tags]*mul,
             [preprocess]*mul)

    if verbose:
        print("Map Phase finished")
        print("Starting Reducing/Inverting into {} partitions".format(partitions.__len__()))

    pool.map(__reduce, partitions)

    if verbose:
        print("Merge Partitions and remove temporary directories")

    with open(output_filepath, 'w') as output_file:
        files = sorted(glob.glob(posting_path+"res"+'*'))
        files_meta = glob.glob(posting_path+"meta"+"*")
        num_documents = 0

        for file in files_meta:
            with open(file, "r") as f:
                print(num_documents)
                for line in f:
                    num_documents += int(line.split("\n")[0])

        output_file.write("{}\n".format(num_documents))

        for file in files:
            with open(file, 'r') as f:
                shutil.copyfileobj(f, output_file)

    files = sorted(glob.glob(posting_path+"doc*"))

    document_length_counter = Counter()
    document_terms_counter = Counter()

    for file in files:
        document_stats = load_document_stats(file)
        document_terms = document_stats['terms']
        document_length = document_stats['length']
        for c in document_terms:
            document_terms_counter[c] += document_terms[c]
            document_length_counter[c] += document_length[c]

    __write_document_stats(document_stats_path, document_terms_counter, document_length_counter)

    __down()


def __map(split, strip_html_tags,strip_html_entities,strip_square_bracket_tags, preprocess):
    generate_tokens_for_files_distributed(split,
                                          strip_html_tags=strip_html_tags,
                                          strip_html_entities=strip_html_entities,
                                          strip_square_bracket_tags=strip_square_bracket_tags,
                                          preprocess=preprocess)


def __reduce(partition):
    segment_path = "./segmented_files/"
    posting_path = "./postings/"

    files = glob.glob(segment_path+partition+"*")

    print("Merge Segmented files of {} partition".format(partition))

    with open(posting_path + partition + "tmp", 'w') as output_file:
        for file in files:
            with open(file, 'r') as f:
                f.readline()
                shutil.copyfileobj(f, output_file)

    print("Sort Merge of {} partition".format(partition))

    with open(posting_path + partition, "w") as output_file:
        f = open(posting_path + partition + "tmp", "r")
        fs = sorted(f)
        output_file.writelines(fs)
        f.close()

    print("reducing {} partition started".format(partition))

    with open(posting_path + partition, "r") as file:
        output_file = open(posting_path+"res"+partition, "w")
        old_key, old_value = file.readline().strip("\n").split(" ")
        posts = [old_value]
        document_terms_counter = Counter()
        document_length_counter = Counter()
        for line in file:
            key, value = line.strip("\n").split(" ")
            if old_key != key:
                __flush_index_entry(output_file, old_key, __to_bag_of_words(posts),
                                    document_terms_counter, document_length_counter)
                old_key = key
                posts = []
            posts.append(value)
        output_file.flush()
        output_file.close()

    __write_document_stats(posting_path + "doc" + partition, document_terms_counter, document_length_counter)

    gc.collect()

    print("reducing {} partition finished".format(partition))


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


def __merge_spimi_blocks(output_file, document_stats_path, block_filepaths):
    block_files = list(map(lambda filepath: open(filepath, 'r'), block_filepaths))

    head_entries = list(map(lambda file: __read_token(file), block_files))
    head_terms = list(map(lambda entry: entry.term, head_entries))

    num_files = len(block_filepaths)
    num_closed = 0

    document_terms_counter = Counter()
    document_length_counter = Counter()

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

        __flush_index_entry(output_file, smallest_term,
                            merged_postings.items(),
                            document_terms_counter, document_length_counter)

    for block_file in block_files:
        if not block_file.closed:
            block_file.close()

    __write_document_stats(document_stats_path,
                           document_terms_counter,
                           document_length_counter)


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


def load_document_stats(filepath):
    """Loads document level stats which were collected
    during index creation
    """

    with open(filepath, 'r') as f:
        return json.loads(f.read())


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


def __flush_index_entry(file, term, postings_list,
                        document_terms_counter, document_length_counter):
    """Collects document stats and write the given index entry to disk
    """

    for document_id, term_frequency in postings_list:
        document_terms_counter[document_id] += 1
        document_length_counter[document_id] += term_frequency

    __write_index_entry(file, term, postings_list)


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


def __write_document_stats(filepath, document_terms_counter,
                           document_length_counter):
    """Writes various document level stats to disk
    """
    stats = {
        'terms': document_terms_counter,
        'length': document_length_counter
    }

    with open(filepath, 'w') as f:
        f.write(json.dumps(stats))


def __to_bag_of_words(words):
    return Counter(words).items()