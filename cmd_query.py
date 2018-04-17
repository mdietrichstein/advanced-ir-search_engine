from preprocessing import split_words, create_preprocessor
from indexing import create_index_reader, load_document_stats
from searching import simple_tfidf_search, cosine_tfidf_search, simple_bm25_search, simple_bm25va_search

import click


@click.group()
@click.option('--query', required=True,
              help='Term to search for')
@click.option('--index_file', required=True, type=click.Path(exists=True),
              help='Path to index file')
@click.option('--stats_file', required=True, type=click.Path(exists=True),
              help='Path to document stats file')
@click.option('--enable_case_folding/--disable_case_folding',
              default=True, show_default=True,
              help='Enable/Disable case folding during preprocessing')
@click.option('--enable_stemmer/--disable_stemmer',
              default=True, show_default=True,
              help='Enable/Disable stemmer during preprocessing')
@click.option('--enable_lemmatizer/--disable_lemmatizer',
              default=False, show_default=True,
              help='Enable/Disable lemmatizer during preprocessing')
@click.option('--enable_remove_stop_words/--disable_remove_stop_words',
              default=True, show_default=True,
              help='Enable/Disable removal of stop words during preprocessing')
@click.option('--min_word_length',
              default=2, show_default=True,
              help='Minimum word length. Words shorter than the given length are ignored')
@click.option('--enable_strip_html_tags/--disable_strip_html_tags',
              default=True, show_default=True,
              help='Enable/Disable removal of html tags')
@click.option('--enable_strip_html_entities/--disable_strip_html_entities',
              default=True, show_default=True,
              help='Enable/Disable removal of html entities, like "&amp;"')
@click.option('--enable_strip_square_bracket_tags/--disable_strip_square_bracket_tags',
              default=True, show_default=True,
              help='Enable/Disable removal of tags in square brackets, like "[BR]"')
@click.pass_context
def cli(ctx, query, index_file, stats_file,
        enable_case_folding, enable_stemmer, enable_lemmatizer,
        enable_remove_stop_words, min_word_length,
        enable_strip_html_tags, enable_strip_html_entities,
        enable_strip_square_bracket_tags):

        def run_eval(ranking_method, params={}):
            preprocess = create_preprocessor(enable_case_folding=enable_case_folding,
                                             enable_remove_stop_words=enable_remove_stop_words,
                                             enable_stemmer=enable_stemmer,
                                             enable_lemmatizer=enable_lemmatizer,
                                             min_length=min_word_length)

            words = split_words(query,
                                strip_html_tags=enable_strip_html_tags,
                                strip_html_entities=enable_strip_html_entities,
                                strip_square_bracket_tags=enable_strip_square_bracket_tags)

            search_terms = preprocess(words)

            click.echo(f'Searching for "{query}" using "{ranking_method}"')
            click.echo(f'Words: "{words}"')
            click.echo(f'Terms: "{search_terms}"')

            click.echo(f'Loading document stats from {stats_file}')
            document_stats = load_document_stats(stats_file)
            click.echo('done')

            click.echo(f'Loading search index from {index_file}')
            click.echo('This might take a while')
            number_of_documents, index_reader_generator = create_index_reader(index_file)
            index_reader = index_reader_generator()
            index = list(index_reader)
            click.echo('done')

            document_scores = None

            if ranking_method == 'tfidf':
                document_scores = simple_tfidf_search(number_of_documents,
                                                      index,
                                                      search_terms)
            elif ranking_method == 'cosine_tfidf':
                document_scores = cosine_tfidf_search(number_of_documents,
                                                      index,
                                                      search_terms)
            elif ranking_method == 'bm25':
                document_scores = simple_bm25_search(number_of_documents,
                                                     index,
                                                     search_terms,
                                                     document_stats,
                                                     k1=params['k1'],
                                                     b=params['b'],
                                                     k3=params['k3'])
            elif ranking_method == 'bm25va':
                document_scores = simple_bm25va_search(number_of_documents,
                                                       index,
                                                       search_terms,
                                                       document_stats,
                                                       k1=params['k1'],
                                                       k3=params['k3'])

            for document_score in document_scores[:50]:
                print(f'{document_score[1]}\t{document_score[0]}')

        ctx.obj['RUNNER'] = run_eval


@cli.command()
@click.pass_context
def tfidf(ctx):
    ctx.obj['RUNNER']('tfidf')


@cli.command()
@click.pass_context
def cosine_tfidf(ctx):
    ctx.obj['RUNNER']('cosine_tfidf')


@cli.command()
@click.option('--k1', default=1.2, show_default=True,
              help='k1 parameter for bm25')
@click.option('--b', default=0.75, show_default=True,
              help='b parameter for bm25')
@click.option('--k3', default=8.0, show_default=True,
              help='k3 parameter for bm25')
@click.pass_context
def bm25(ctx, k1, b, k3):
    ctx.obj['RUNNER']('bm25', { 'k1': k1, 'b': b, 'k3': k3})


@cli.command()
@click.option('--k1', default=1.2, show_default=True,
              help='k1 parameter for bm25va')
@click.option('--k3', default=8.0, show_default=True,
              help='k3 parameter for bm25va')
@click.pass_context
def bm25va(ctx, k1, k3):
    ctx.obj['RUNNER']('bm25va', { 'k1': k1, 'k3': k3})


if __name__ == '__main__':
    cli(obj={})
