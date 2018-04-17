## Prerequisites

This project requires at least python 3.6.

### Install Dependencies

`pip install -r requirements.txt`


## Index Creation

### Run

Run `python cmd_index.py --help` for instructions.

#### Example:

To create an index and document stats using the SPIMI method, run:
`python cmd_index.py --document_folder=./data/TREC8all/Adhoc/ --index_file=spimi.index --stats_file=spimi.stats spimi`

### Output

The script creates two output files:

* `index_file`: Inverted index
* `stats_file`: Document stats collected during index creation (document lengths and term counts)

### Index Format

The index file format is text-based.

The first line states the number of unique documents in the index.

Each consecutive line represents a term including related data:

`<TERM> <DOCUMENT_FREQUENCY> <POSTINGS>`

* TERM - The term itself
* DOCUMENT_FREQUENCY - Document Frequency (Number of documents the term appears in)
* POSTINGS - A comma-separated list of documents the term appears in
  along with the term frequency separated by pipe in the given document:
  <DOCUMENT_ID>|<TERM_FREQUENCY>,<DOCUMENT_ID>|<TERM_FREQUENCY>,...
* TERM_FREQUENCY - Number of times the term appears in the corresponding document 

## Evaluation

### Run

Run `python cmd_search.py --help` for instructions.

#### Example:

To evaluate a topic list on a previously created SPIMI index using a bm25 ranking, run: `python cmd_search.py --output_file=out.txt --run_name=dev --topics_file=./data/TREC8all/topicsTREC8Adhoc.txt --index_file=spimi.index --stats_file=spimi.stats bm25`

### Output

The script creates an output file which can be used with `trec_eval`, like: `trec_eval -q -m map -c ./data/TREC8all/qrels.trec8.adhoc.parts1-5 ./out.txt`