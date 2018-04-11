## Prerequisites

### Install Dependencies

`pip install -r requirements.txt`


## Index Creation

###  Configure
Add a bunch of docs to `document_files` in `cmd_index.py`, e.g. `document_files = ['./data/TREC8all/Adhoc/latimes/la010189']`

### Run

`python cmd_index.py`

### Output

The script creates the following output files:

* `spimi.index`: Index created by using the SPIMI method
* `spimi.stats`: Document stats collected during spimi index creation

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

`python cmd_evaluate.py`

### Output

The script creates three result files, one for each ranking algorithm (tf-idf, bm25, bm25va):

* `tfidf_results.txt`
* `bm25_results.txt`
* `bm25va_results.txt`

Use `trec_eval` to evaluate the results.
