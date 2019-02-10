# Advanced Information Retrieval - Search Engine

## Prerequisites

The project requires at least `Python 3.6`.

Run `pip install -r requirements.txt` to install the required dependencies

## Usage

### Index Creation

The `indexer.sh` command is responsible for creating an index and collecting document statistics from a given corpus.

It supports two different index creation methods: 

* `simple`: Index creation via a simple in-memory posting list
* `spimi`: Index creation via the _Single Pass in Memory Index_ method

#### Example Usage

```bash
./indexer.sh --document_folder=./data/TREC8all/Adhoc/ --index_file=spimi.index --stats_file=spimi.stats spimi
```

#### CLI Options

Various command line parameters allow the customization of tokenization and preprocessing options.

```bash
Usage: indexer.sh [OPTIONS] COMMAND [ARGS]...

Options:
  --document_folder PATH          Path to the folder which contains the
                                  documents to be indexed  [required]
  --index_file TEXT               Output filename for index file  [required]
  --stats_file TEXT               Output filename for document stats file
                                  [required]
  --enable_case_folding / --disable_case_folding
                                  Enable/Disable case folding during
                                  preprocessing  [default: True]
  --enable_stemmer / --disable_stemmer
                                  Enable/Disable stemmer during preprocessing
                                  [default: True]
  --enable_lemmatizer / --disable_lemmatizer
                                  Enable/Disable lemmatizer during
                                  preprocessing  [default: False]
  --enable_remove_stop_words / --disable_remove_stop_words
                                  Enable/Disable removal of stop words during
                                  preprocessing  [default: True]
  --min_word_length INTEGER       Minimum word length. Words shorter than the
                                  given length are ignored  [default: 2]
  --enable_strip_html_tags / --disable_strip_html_tags
                                  Enable/Disable removal of html tags
                                  [default: True]
  --enable_strip_html_entities / --disable_strip_html_entities
                                  Enable/Disable removal of html entities,
                                  like "&amp;"  [default: True]
  --enable_strip_square_bracket_tags / --disable_strip_square_bracket_tags
                                  Enable/Disable removal of tags in square
                                  brackets, like "[BR]"  [default: True]

Commands:
  simple
  spimi
  map_reduce
```

The `spimi` subcommand supports the following additional command-line options:
```bash
  --max_tokens_per_block INTEGER  Maximum number of tokens allowed in a single spimi block  [default: 10000000]
```

The `map_reduce` subcommand supports the following additional command-line options:
```bash
  --blocksize INTEGER  Size (in Megabyte) of documents a process should take one at a time (during Map Phase)
  --num_nodes INTEGER  Number of Nodes/Processes over which the taskload will be distributed. Will typically default to the number of available cores
```

### Trec Evaluation Metrics Creation
The `searcher.sh` command is responsible for searching an index using topic files. It outputs a result file which is compatible with `trec_eval` and can be used to evaluate the search engine's performance.

It supports the following ranking methods 

* `tfidf`:  Simple sum of term-wise TF-IDF scores (`Overlap Score Measure`)
* `cosine_tfidf`: TF-IDF Vector Space Model score
* `bm25`: Okapi BM25 score 
* `bm25va`: Score calculation using the `Verboseness Fission for BM25 Document Length Normalization` approach

#### Example Usage

```bash
/searcher.sh --output_file=bm25va_results.txt --run_name=bm25va --topics_file=./data/TREC8all/topicsTREC8Adhoc.txt --index_file=spimi.index --stats_file=spimi.stats bm25va
```

#### CLI Options

```bash
Usage: searcher.sh [OPTIONS] COMMAND [ARGS]...

Options:
  --output_file TEXT              Output filename for results file  [required]
  --run_name TEXT                 Name for run in results file  [required]
  --topics_file PATH              Path to a file containing search topics
                                  [required]
  --index_file PATH               Path to index file  [required]
  --stats_file PATH               Path to document stats file  [required]
  --enable_case_folding / --disable_case_folding
                                  Enable/Disable case folding during
                                  preprocessing  [default: True]
  --enable_stemmer / --disable_stemmer
                                  Enable/Disable stemmer during preprocessing
                                  [default: True]
  --enable_lemmatizer / --disable_lemmatizer
                                  Enable/Disable lemmatizer during
                                  preprocessing  [default: False]
  --enable_remove_stop_words / --disable_remove_stop_words
                                  Enable/Disable removal of stop words during
                                  preprocessing  [default: True]
  --min_word_length INTEGER       Minimum word length. Words shorter than the
                                  given length are ignored  [default: 2]
  --enable_strip_html_tags / --disable_strip_html_tags
                                  Enable/Disable removal of html tags
                                  [default: True]
  --enable_strip_html_entities / --disable_strip_html_entities
                                  Enable/Disable removal of html entities,
                                  like "&amp;"  [default: True]
  --enable_strip_square_bracket_tags / --disable_strip_square_bracket_tags
                                  Enable/Disable removal of tags in square
                                  brackets, like "[BR]"  [default: True]
  --help                          Show this message and exit.

Commands:
  bm25
  bm25va
  cosine_tfidf
  tfidf
```


The `bm25` subcommand supports the following additional command-line options:

```bash
  --k1 FLOAT  k1 parameter for bm25  [default: 1.2]
  --b FLOAT   b parameter for bm25  [default: 0.75]
  --k3 FLOAT  k3 parameter for bm25  [default: 8.0]
```

The `bm25va` subcommand supports the following additional command-line options:

```bash
  --k1 FLOAT  k1 parameter for bm25va  [default: 1.2]
  --k3 FLOAT  k3 parameter for bm25va  [default: 8.0]
```

## Index Structure

The index file format is text-based.

The first line states the number of unique documents in the index.

Each consecutive line represents a term including related data:

`<TERM> <DOCUMENT_FREQUENCY> <POSTINGS>`

* `TERM` - The term itself
* `DOCUMENT_FREQUENCY` - Document Frequency (Number of documents the term appears in)
* `POSTINGS` - A comma-separated list of documents the term appears in
  along with the term frequency separated by pipe in the given document:
  `<DOCUMENT_ID>|<TERM_FREQUENCY>,<DOCUMENT_ID>|<TERM_FREQUENCY>,...`
* `TERM_FREQUENCY` - Number of times the term appears in the corresponding document 


**Index Improvements**

A potential improvement to the current index structure would be to include pre-calculated TF and IDF scores to the index.  This was not implemented in the current version of the search engine since the performance cost of search-time TF-IDF calculation is negligible, whereas index creation is already very time consuming.

**Documents Statistics**

The indexer script also creates a json file containing the length and number of unique terms of each document. This data is used during the calculation of BM25 and BM25VA scores.

## Ranking Method Performance Comparisons

### Setup

The comparison has been conducted on an index created with the following parameters:
```bash
./indexer.sh --document_folder=./data/TREC8all/Adhoc/ --index_file=spimi.index --stats_file=spimi.stats --enable_case_folding --enable_stemmer --disable_lemmatizer --enable_remove_stop_words --min_word_length=2 --enable_strip_html_tags --enable_strip_html_entities --enable_strip_square_bracket_tags spimi --max_tokens_per_block=10000000
```

Search results have been created with the following parameters

```bash
./searcher.sh --output_file=<OUTPUT_FILE> --run_name=<RUN_NAME> --topics_file=./data/TREC8all/topicsTREC8Adhoc.txt --index_file=spimi.index --stats_file=spimi.stats --enable_case_folding --enable_stemmer --disable_lemmatizer --enable_remove_stop_words --min_word_length=2 --enable_strip_html_tags --enable_strip_html_entities --enable_strip_square_bracket_tags <RANKING_METHOD>
```

The parameters for BM25 and BM25VA have been left as is (default values).


### Ranking Method Scores

|         | TF-IDF Overlap | TF-IDF VSM | BM25   | BM25VA |
| ------: | :------------: | :--------: | :----: | :----: |
| **401** | 0.0003         |  0.0000    | 0.0003 | 0.0000 |
| **402** | 0.0000         |  0.0246    | 0.1239 | 0.1267 |
| **403** | 0.1408         |  0.3019    | 0.7406 | 0.6736 |
| **404** | 0.0072         |  0.0255    | 0.0539 | 0.0712 |
| **405** | 0.0091         |  0.0327    | 0.1004 | 0.1200 |
| **406** | 0.0258         |  0.1906    | 0.2036 | 0.2048 |
| **407** | 0.0302         |  0.2171    | 0.1129 | 0.1294 |
| **408** | 0.0116         |  0.0163    | 0.0771 | 0.0810 |
| **409** | 0.0018         |  0.0665    | 0.1516 | 0.1344 |
| **410** | 0.0032         |  0.5059    | 0.6045 | 0.5308 |
| **411** | 0.0138         |  0.1562    | 0.1375 | 0.1526 |
| **412** | 0.0000         |  0.0000    | 0.0104 | 0.0169 |
| **413** | 0.0280         |  0.0033    | 0.0889 | 0.0708 |
| **414** | 0.0058         |  0.0364    | 0.1742 | 0.1294 |
| **415** | 0.0976         |  0.1935    | 0.2457 | 0.2517 |
| **416** | 0.0960         |  0.0329    | 0.2282 | 0.2212 |
| **417** | 0.0102         |  0.0002    | 0.1682 | 0.2048 |
| **418** | 0.0184         |  0.0741    | 0.0789 | 0.0973 |
| **419** | 0.0463         |  0.0159    | 0.0578 | 0.0580 |
| **420** | 0.1004         |  0.1804    | 0.3085 | 0.3126 |
| **421** | 0.0025         |  0.0009    | 0.0189 | 0.0402 |
| **422** | 0.0333         |  0.0075    | 0.1250 | 0.0997 |
| **423** | 0.6140         |  0.6044    | 0.6263 | 0.6713 |
| **424** | 0.0110         |  0.0064    | 0.0386 | 0.0227 |
| **425** | 0.0182         |  0.0194    | 0.1952 | 0.1526 |
| **426** | 0.0000         |  0.0010    | 0.0025 | 0.0030 |
| **427** | 0.0511         |  0.1196    | 0.1550 | 0.1747 |
| **428** | 0.0053         |  0.0040    | 0.0277 | 0.0280 |
| **429** | 0.1288         |  0.2735    | 0.4234 | 0.3139 |
| **430** | 0.5000         |  0.2460    | 0.5660 | 0.5080 |
| **431** | 0.1131         |  0.0230    | 0.2207 | 0.2460 |
| **432** | 0.0036         |  0.0012    | 0.0064 | 0.0097 |
| **433** | 0.0000         |  0.1141    | 0.0027 | 0.0119 |
| **434** | 0.0573         |  0.0207    | 0.1252 | 0.1203 |
| **435** | 0.0081         |  0.0024    | 0.0121 | 0.0156 |
| **436** | 0.0001         |  0.0038    | 0.0433 | 0.0336 |
| **437** | 0.0142         |  0.0208    | 0.0072 | 0.0089 |
| **438** | 0.0327         |  0.0006    | 0.0498 | 0.0617 |
| **439** | 0.0002         |  0.0023    | 0.0043 | 0.0033 |
| **440** | 0.0000         |  0.0072    | 0.0057 | 0.0093 |
| **441** | 0.1245         |  0.5003    | 0.4378 | 0.4746 |
| **442** | 0.0000         |  0.0117    | 0.0023 | 0.0128 |
| **443** | 0.0158         |  0.0051    | 0.0225 | 0.0281 |
| **444** | 0.0947         |  0.8960    | 0.4678 | 0.4112 |
| **445** | 0.0000         |  0.0082    | 0.0912 | 0.0653 |
| **446** | 0.0026         |  0.0000    | 0.0084 | 0.0155 |
| **447** | 0.1063         |  0.2851    | 0.4698 | 0.2540 |
| **448** | 0.0000         |  0.0000    | 0.0116 | 0.0181 |
| **449** | 0.0371         |  0.0194    | 0.0467 | 0.0691 |
| **450** | 0.0150         |  0.0760    | 0.1196 | 0.0945 |
| **all** | **0.0527**     | **0.1071**  | **0.1600** | **0.1513** |

### Statistical Significance

The table below shows the p-values obtained by running a t-test comparing the per-topic scores of the respective ranking methods. Significant p-values (lower than 0.05) have been highlighted.

|                | TF-IDF Overlap | TF-IDF VSM | BM25 | BM25VA |
| -------------- | :------------: | :--------: | :---: | :----: |
| **TF-IDF Overlap** | 1.00000 | 0.07657 | **0.00080** | **0.00098** |
| **TF-IDF VSM**     | 0.07657 | 1.00000 | 0.15642 | 0.21357 |
| **BM25**           | **0.00080** | 0.15642 | 1.00000 | 0.81196 |
| **BM25VA**         | **0.00098** | 0.21357 | 0.81196 | 1.00000 |

### Discussion

The trec_eval MAP scores suggest that the BM25 variants perform best, followed by TF-IDF VSM and (with a distance) TF-IDF Overlap.

The poor performance of TF-IDF Overlap was expected since it is very sensitive to document length. Using a ranking method which has been designed with this problem in mind (like all other methods) immediately improves the evaluation results.

BM25 based methods are able to obtain the highest scores which was also expected since probabilistic ranking methods have proven to perform best in real world applications.


The statistical tests have shown that the former statements can not be considered with certainty though. The only statement that can be made with high confidence is that the TF-IDF Overlap model performed significantly worse than it's BM25 counterparts.

The experiment was also unable to reproduce the results shown in _Verboseness Fission for BM25 Document Length Normalization_. This may be attributed to the possibility that the scope/verboseness problem is not relevant to the document corpus that has been used for evaluation.

Another more obvious reason could be implementation issues or different parameters during preprocessing, tokenization, ...



## BM25VA Score Calculation

Configuration: `k1=1.2, k=8.0`

Search Query: "Gorbachev Yeltsin"
Search Terms after preprocessing: ['gorbachev', 'yeltsin']

Search results:
```
19.209480277622937      LA111490-0115
19.151293832254865      LA053090-0023
...
```

### Corpus Statistics


* Number of documents (`N`): `523951`
* Average document length (`avgdl`): `275.79141339707076`
* Mean average  term frequency (`mavgft`): `1.5089422117484923`

### Document level statistics


|                   | Number of unique terms (`T`) | Document length (`D`) |
| ----------------- | :----------------------: | :---------------: |
| **LA111490-0115** | 330                    | 583             |
| **LA053090-0023** | 515                    | 913             |


### Terms

|               | Term Frequency in query (`tfq`) | Document frequency (`dft`) | Term Frequency in Document (`tfd`) - LA111490-0115 | Term Frequency in Document (`tfd`) - LA053090-0023 |
| ------------- | :-----------------------------: | :------------------------: | :--------------------------------------------------: | :--------------------------------------------------: |
| **gorbachev** |                1                |            2769            | 20                                                 | 26                                                 |
| **yeltsin**   |                1                |            7314            | 21                                                 | 24                                                 |


### BM25VA Formula

$$
Bva = \frac{1}{ (mavgft^2)} * \frac{D}{T} + (1 - \frac{1}{mavgft}) * \frac{D}{avgdl}
$$

$$
Score =  \frac{(k3 + 1) * tfq}{k3 + tfq} * \frac{(k1 + 1) * tfd}{(k1 * Bva) + tfd} * log(\frac{N - dft + 0.5}{dft + 0.5})
$$

### Results

|           | Score - LA111490-0115 | Score - LA053090-0023 |
| --------- | :-------------------: | :-----------------------: |
| **gorbachev** | 10.577431342458835 | 10.595562886478845 |
| **yeltsin**   | 8.632048935164104 | 8.55573094577602 |
| **Final Score** | **19.209480277622937** | **19.151293832254865** |

