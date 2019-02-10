# README

A document search engine written from scratch in Python. Based on concepts from Stanford's [Introduction to Information Retrieval Book](https://nlp.stanford.edu/IR-book/).

Uses the TREC8Adhoc part of the [TIPSTER collection](https://catalog.ldc.upenn.edu/LDC93T3A) for index building and evaluation. You'll have to obtain `TREC8Adhoc.tar.bz2` from this collection ([Disk 4 & 5](https://trec.nist.gov/data/qa/T8_QAdata/disks4_5.html)) to reproduce the [reported results](https://github.com/mdietrichstein/advanced-ir-search_engine/blob/master/REPORT.md).

To evaluate the results with `trec_eval` you will have to download the [TREC-8 ad hoc qrels](https://trec.nist.gov/data/qrels_eng/qrels.trec8.adhoc.parts1-5.tar.gz).

Features:
* Inverted index construction methods: Simple, Single-pass in-memory indexing (SPIMI )and Map Reduce
* Document similarity metrics: TF-IDF, BM25, [BM25VA (PDF Link)](https://publik.tuwien.ac.at/files/PubDat_244472.pdf) and TF-IDF Cosine Distance
* Performance evaluation on TREC and result reporting in [qrel format](https://trec.nist.gov/data/qrels_eng/).

An early Rust port is located at [ir-search-engine-rust](https://github.com/mdietrichstein/ir-search-engine-rust).
### Prerequisites

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
