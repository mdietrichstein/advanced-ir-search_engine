from preprocessing import create_preprocessor
from indexing import create_index_simple, create_index_spimi, create_index_map_reduce

import os
import glob
import nltk
import click


@click.group()
@click.option('--document_folder', required=True, type=click.Path(exists=True),
              help='Path to the folder which contains the documents to be indexed')
@click.option('--index_file', required=True,
              help='Output filename for index file')
@click.option('--stats_file', required=True,
              help='Output filename for document stats file')
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
def cli(ctx, document_folder, index_file, stats_file,
        enable_case_folding, enable_stemmer, enable_lemmatizer,
        enable_remove_stop_words, min_word_length,
        enable_strip_html_tags, enable_strip_html_entities,
        enable_strip_square_bracket_tags):
    nltk.download('wordnet')

    preprocessor = create_preprocessor(enable_case_folding=enable_case_folding,
                                       enable_remove_stop_words=enable_remove_stop_words,
                                       enable_stemmer=enable_stemmer,
                                       enable_lemmatizer=enable_lemmatizer,
                                       min_length=min_word_length)

    glob_pattern = document_folder + '/**'
    document_files = [fname for fname in glob.glob(glob_pattern, recursive=True) if os.path.isfile(fname)]

    click.echo()
    click.echo('Processing {} file(s)'.format(len(document_files)))
    click.echo()

    ctx.obj['INDEX_FILE'] = index_file
    ctx.obj['STATS_FILE'] = stats_file
    ctx.obj['DOCUMENT_FILES'] = document_files
    ctx.obj['PREPROCESSOR'] = preprocessor

    ctx.obj['STRIP_HTML_TAGS'] = enable_strip_html_tags
    ctx.obj['STRIP_HTML_ENTITIES'] = enable_strip_html_entities
    ctx.obj['STRIP_SQUARE_BRACKET_TAGS'] = enable_strip_square_bracket_tags


@cli.command()
@click.pass_context
def simple(ctx):
    preprocessor = ctx.obj['PREPROCESSOR']
    click.echo(f'Writing index using simple method to {ctx.obj["INDEX_FILE"]} and document stats to {ctx.obj["STATS_FILE"]}')
    click.echo('Reading source files')

    create_index_simple(ctx.obj['DOCUMENT_FILES'], preprocessor,
                        ctx.obj['INDEX_FILE'],
                        ctx.obj['STATS_FILE'],
                        strip_html_tags=ctx.obj['STRIP_HTML_TAGS'],
                        strip_html_entities=ctx.obj['STRIP_HTML_ENTITIES'],
                        strip_square_bracket_tags=ctx.obj['STRIP_SQUARE_BRACKET_TAGS'])


@cli.command()
@click.option('--max_tokens_per_block', default=10000000, show_default=True,
              help='Maximum number of tokens allowed in a single spimi block')
@click.pass_context
def spimi(ctx, max_tokens_per_block):
    preprocessor = ctx.obj['PREPROCESSOR']
    click.echo(f'Writing index using spimi method to {ctx.obj["INDEX_FILE"]} and document stats to {ctx.obj["STATS_FILE"]}')
    click.echo('Reading source files')

    create_index_spimi(ctx.obj['DOCUMENT_FILES'], preprocessor,
                       ctx.obj['INDEX_FILE'],
                       ctx.obj['STATS_FILE'],
                       max_tokens_per_block=max_tokens_per_block,
                       strip_html_tags=ctx.obj['STRIP_HTML_TAGS'],
                       strip_html_entities=ctx.obj['STRIP_HTML_ENTITIES'],
                       strip_square_bracket_tags=ctx.obj['STRIP_SQUARE_BRACKET_TAGS'])


@cli.command()
@click.option('--blocksize', default=16, show_default=True,
              help='Size of data the Map Process takes one at a time in Megabyte')
@click.option('--num_nodes', default=None, show_default=True,
              help='Number of Processes over which the work load is distributed. Typically defaults to the number of cores')
@click.pass_context
def map_reduce(ctx, blocksize, num_nodes):
    preprocessor = ctx.obj['PREPROCESSOR']
    click.echo(f'Writing index using map_redduce to {ctx.obj["INDEX_FILE"]} and document stats to {ctx.obj["STATS_FILE"]}')
    click.echo('Reading source files')

    create_index_map_reduce(ctx.obj['DOCUMENT_FILES'], preprocessor,
                        ctx.obj['INDEX_FILE'],
                        ctx.obj['STATS_FILE'],
                        strip_html_tags=ctx.obj['STRIP_HTML_TAGS'],
                        strip_html_entities=ctx.obj['STRIP_HTML_ENTITIES'],
                        strip_square_bracket_tags=ctx.obj['STRIP_SQUARE_BRACKET_TAGS'],
                        blocksize=blocksize,
                        num_nodes=num_nodes)


if __name__ == '__main__':
    cli(obj={})