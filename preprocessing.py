import re
import Stemmer
from nltk.stem import WordNetLemmatizer

STEMMER = Stemmer.Stemmer('english')
LEMMATIZER = WordNetLemmatizer()
HTML_TAG_PATTERN = re.compile(r'<.*?>')
HTML_ENTITY_PATTERN = re.compile('&[a-zA-Z][-.a-zA-Z0-9]*[^a-zA-Z0-9]')
SQUARE_BRACKET_TAG_PATTERN = re.compile(r'\[.*?\]')

SPLIT_WORDS_PATTERN = re.compile(r'\s|\.|\:|\?|\(|\)|\[|\]|\{|\}|\<|\>|\'|\!|\"|\-|,|;|\$|\*|\%|#')

# From https://www.textfixer.com/tutorials/common-english-words.txt via https://en.wikipedia.org/wiki/Stop_words
STOP_WORDS = {'a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your'}


def split_words(text,
                strip_html_tags=True,
                strip_html_entities=True,
                strip_square_bracket_tags=True):

    """Takes an input text an generates a list of words. Removes certain words
    specified by this method's 'strip_' arguments
    """

    if strip_html_tags:
        text = re.sub(HTML_TAG_PATTERN, '', text)

    if strip_html_entities:
        text = re.sub(HTML_ENTITY_PATTERN, '', text)

    if strip_square_bracket_tags:
        text = re.sub(SQUARE_BRACKET_TAG_PATTERN, '', text)

    return list(filter(lambda word: word != '', SPLIT_WORDS_PATTERN.split(text)))


def create_preprocessor(enable_case_folding=True,
                        enable_remove_stop_words=True,
                        enable_stemmer=True,
                        enable_lemmatizer=False,
                        min_length=2):

    """Generates a preprocessing function configured to apply the specified
    processing steps
    """
    steps = []

    if enable_case_folding:
        steps.append(__case_folding)

    if enable_remove_stop_words:
        steps.append(__remove_stop_words)

    if enable_stemmer:
        steps.append(__stem)

    if enable_lemmatizer:
        steps.append(__lemmatize)

    if min_length:
        steps.append(lambda words: __remove_short_words(words, min_length))

    def fn_preprocess(words):
        words = list(words)

        for i, step in enumerate(steps):
            words = list(step(words))

        return words

    return fn_preprocess


def __case_folding(words):
    return map(lambda word: word.casefold(), words)


def __remove_stop_words(words):
    return filter(lambda word: word not in STOP_WORDS, words)


def __stem(words):
    return map(lambda word: STEMMER.stemWord(word), words)


def __lemmatize(words):
    return map(lambda word: LEMMATIZER.lemmatize(word), words)


def __remove_short_words(words, min_length):
    return filter(lambda word: len(word) >= min_length, words)
