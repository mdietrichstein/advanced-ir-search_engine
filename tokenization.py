import re
from tqdm import tqdm
from xml.dom import minidom
from pathlib import Path

from preprocessing import split_words, create_preprocessor

DOC_PATTERN = re.compile(r'<DOC>(.*?)<\/DOC>', re.DOTALL | re.M)
DOCNO_PATTERN = re.compile(r'<DOCNO>(.*?)<\/DOCNO>', re.DOTALL | re.M)
TEXT_PATTERN = re.compile(r'<TEXT>(.*?)<\/TEXT>', re.DOTALL | re.M)


def generate_tokens_for_files(filepaths, encoding='latin-1',
                              use_regex_parser=True,
                              strip_html_tags=True,
                              strip_html_entities=True,
                              strip_square_bracket_tags=True,
                              preprocess=create_preprocessor()):
    """Generator which provides a list of (doc_id, term) pairs for documents
    contained in the given files
    """
    for filepath in tqdm(filepaths, total=len(filepaths)):
        documents = None

        if use_regex_parser:
            documents = __regex_parse_documents_from_file(filepath)
        else:
            documents = __xml_parse_documents_from_file(filepath)

        for document in documents:
            (doc_id, content) = document

            words = split_words(content,
                                strip_html_tags=strip_html_tags,
                                strip_html_entities=strip_html_entities,
                                strip_square_bracket_tags=strip_square_bracket_tags)

            terms = preprocess(words)

            for term in terms:
                yield (doc_id, term)


def __regex_parse_documents_from_file(file_path, encoding='latin-1'):
    """Loads all documents from the given SGML file
    using REGEX and returns them
    """

    # read whole file
    content = Path(file_path, encoding=encoding).read_text()

    documents = []

    for doc in DOC_PATTERN.findall(content):
        doc_number = DOCNO_PATTERN.findall(doc)[0].strip()
        text = TEXT_PATTERN.findall(doc)

        if not text:
            continue  # ignore documents without text

        text = text[0].strip()
        documents.append((doc_number, text))

    return documents


def __xml_parse_documents_from_file(file_path, encoding='latin-1'):
    """Loads all documents from the given SGML file
    using a XML parser and returns them
    """

    # read whole file
    content = Path(file_path, encoding=encoding).read_text()

    # split individual root level components to allow xml parsing
    doc_strings = content.split('</DOC>')

    documents = []

    for doc_string in doc_strings:
        if doc_string.strip() == '':
            continue

        # we have to add the split word back in to create a valid xml document
        doc_string += '</DOC>'
        xml = minidom.parseString(doc_string)

        # use doc number as doc id
        doc = xml.getElementsByTagName('DOC')[0]

        text_elements = doc.getElementsByTagName('TEXT')

        if not text_elements:
            continue  # ignore documents without text

        text = (text_elements[0]).toprettyxml().strip()
        doc_number = __get_xml_text(doc.getElementsByTagName('DOCNO')[0]).strip()

        documents.append((doc_number, text))

    return documents


def __get_xml_text(node):
    """Finds all child text nodes of the given xml node and returns them
    concatenated as a string
    """
    result = ""
    for node in node.childNodes:
        if node.nodeType == node.TEXT_NODE:
            result += node.data

    return result
