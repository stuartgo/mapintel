from src.utils import CorpusPreprocess
import os
import sys
import logging
import multiprocessing
from pathlib import Path
from string import punctuation
from nltk.corpus import stopwords
import collections
from gensim import models
import pandas as pd

# https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
# https://radimrehurek.com/gensim/models/doc2cvec.html
# https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html

# Finding project_dir
project_dir = Path(__file__).resolve().parents[2]
data_file = os.path.join(project_dir, "data", "processed", "newsapi_docs.csv")
output_dir = os.path.join(project_dir, "models")


def main():
    logger = logging.getLogger(__name__)

    logger.info('Reading data...')
    # Reading data into memory
    df = pd.read_csv(data_file, names=['id', 'col', 'category', 'text', 'split'])
    logger.info(f'Read data has a size of {df.memory_usage().sum()//1000}Kb')
    original_text = df['text'].copy()
    X_train, X_test = df.loc[df['split'] == 'train', 'text'], df.loc[df['split']=='test', 'text']

    logger.info('Preprocessing data...')
    # Preprocessing text
    prep = CorpusPreprocess(stop_words=stopwords.words('english'), lowercase=True, strip_accents=True,
                            strip_punctuation=punctuation, stemmer=True, max_df=1.0, min_df=1)
    df.loc[df['split']=='train', 'text'] = prep.fit_transform(X_train, tokenize=False)
    df.loc[df['split']=='test', 'text'] = prep.transform(X_test, tokenize=False) 

    # Formatting data
    alldocs = [NewsDocument([tag], row['id'], row['col'], row['category'], row['text'].split(' '), row['split'])
               for tag, (_, row) in enumerate(df.iterrows())]
    del df, X_train, X_test
    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    logger.info('Percentage of documents from test set: {0:.2f}%'.format((len(test_docs)/len(alldocs)*100)))

    # Doc2Vec models
    assert models.doc2vec.FAST_VERSION > -1, "Gensim won't use a C compiler, which will severely increase running time."
    for model in simple_models:
        model.build_vocab(train_docs)
        logger.info("%s vocabulary scanned & state initialized" % model)

    # Training and saving the models
    for model in simple_models:
        logger.info("Training %s..." % model)
        model.train(train_docs, total_examples=len(train_docs), epochs=model.epochs)
        logger.info("Saving %s..." % model)
        model_name = str(model).lower().translate(str.maketrans('', '', punctuation))
        model.save(os.path.join(output_dir, f"{model_name}.model"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    common_kwargs = dict(
        vector_size=100, epochs=20, min_count=2,
        sample=0, workers=multiprocessing.cpu_count(), negative=5, hs=0,
    )

    simple_models = [
        # PV-DBOW plain
        models.doc2vec.Doc2Vec(dm=0, **common_kwargs),
        # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
        models.doc2vec.Doc2Vec(dm=1, window=10, alpha=0.05, comment='alpha=0.05', **common_kwargs),
        # PV-DM w/ concatenation - big, slow, experimental mode
        # window=5 (both sides) approximates paper's apparent 10-word total window size
        models.doc2vec.Doc2Vec(dm=1, dm_concat=1, window=5, **common_kwargs),
    ]

    # Data structure for holding data for each document
    NewsDocument = collections.namedtuple('NewsDocument', ['tags', 'id', 'col', 'category', 'words', 'split'])

    main()