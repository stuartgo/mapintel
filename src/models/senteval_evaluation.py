"""
Evaluates a set of document embeddings using the SentEval pipeline.
SentEval is a toolkit for evaluating the quality of universal general-purpose
sentence representations through a variety of downstream tasks selected from 
community consensus.
Set of document embeddings evaluated:
    - BOW trained on news corpus
    - TF-IDF trained on news corpus 
    - Pre-trained Glove embeddings: http://nlp.stanford.edu/data/glove.840B.300d.zip
    - Pre-trained FastText embeddings: https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
    - Pre-trained Word2vec embeddgins: https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
    - A Simple but Tough-to-Beat Baseline for Sentence Embedding, Arora et al., 2016 with Glove pre-trained embeddings
"""
import io
import json
import logging
import os

import numpy as np
import senteval
from gensim import models
from joblib import load
from sklearn.decomposition import TruncatedSVD
from src import PROJECT_ROOT
from wordfreq import word_frequency


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_wordvec(path_to_vec, word2id):
    """Opens pre-trained word embeddings file and extracts vectors for 
     each word.

    Args:
        path_to_vec (str): path to pre-trained word embeddings
        word2id (dict): dictionary mapping each word to each id

    Returns:
        dict: dictionary mapping each word to the each embedding vector
    """
    # Open path_to_vec and get vectors for each row
    word_vec = {}
    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    # Get dimensionality of embedding vectors
    wvec_dim = list(word_vec.values())[0].shape[0]

    logging.info(
        f'Found {len(word_vec)} words with word vectors, out of {len(word2id)} words')
    return word_vec, wvec_dim


def create_dictionary(sentences, min_threshold=0):
    """Creates a dictionary of all words in the corpus sentences.
    Excludes words with counts less than min_threshold.

    Args:
        sentences (list): list of all sentences from the tranfer task
        min_threshold (int, optional): minimum count for a word to be
         added to the dictionary. Defaults to 0.

    Returns:
        tuple: word frequency dict, id2word list, word2id dict
    """
    # Get counts of each word in corpus
    words_count = {}
    for s in sentences:
        for word in s:
            words_count[word] = words_count.get(word, 0) + 1

    # Exclude words with less count than min_threshold
    if min_threshold > 0:
        newwords_count = {}
        for word in words_count:
            if words_count[word] >= min_threshold:
                newwords_count[word] = words_count[word]
        words_count = newwords_count

    # Getting word relative frequencies
    total = sum(words_count.values())
    words_freq = {k: v / total for k, v in words_count.items()}

    # Set high frequency for these tokens
    words_freq['<s>'] = 1e9 + 4
    words_freq['</s>'] = 1e9 + 3
    words_freq['<p>'] = 1e9 + 2

    # Sort word frequencies pairs according to frequency
    sorted_words = sorted(words_freq.items(),
                          key=lambda x: -x[1])  # inverse sort

    # Get id -> word and word -> id mappings
    id2word, word2id = [], {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return words_freq, id2word, word2id


def main():
    logger = logging.getLogger(__name__)

    evaluators = {}  # where we save each SE (Sentence Evaluator) object

    # BOW sentence embeddings ----------------------------------------------------------------------------------------------

    # Set prepare and batcher functions
    def prepare_bow(params, samples):
        params.count_vect = load(BOW_FILE)
        params.word2id = params.count_vect.vocabulary_
        params.wvec_dim = len(params.word2id)

    def batcher_bow(params, batch):
        batch = [" ".join(text) for text in batch]
        prep_batch = params.count_vect.transform(batch).toarray()
        return prep_batch

    # Add SE to evaluators
    evaluators["BOW"] = senteval.engine.SE(params, batcher_bow, prepare_bow)

    # TF-IDF sentence embeddings -------------------------------------------------------------------------------------------

    # Set prepare and batcher functions
    def prepare_tfidf(params, samples):
        params.tfidf_vect = load(TFIDF_FILE)
        params.word2id = params.tfidf_vect.vocabulary_
        params.wvec_dim = len(params.word2id)

    def batcher_tfidf(params, batch):
        batch = [" ".join(text) for text in batch]
        prep_batch = params.tfidf_vect.transform(batch).toarray()
        return prep_batch

    # Add SE to evaluators
    evaluators["TFIDF"] = senteval.engine.SE(
        params, batcher_tfidf, prepare_tfidf)

    # Glove sentence embeddings --------------------------------------------------------------------------------------------

    # Set prepare and batcher functions
    def prepare_glove(params, samples):
        _, _, params.word2id = create_dictionary(samples)
        params.word_vec, params.wvec_dim = get_wordvec(
            GLOVE_FILE, params.word2id)

    def batcher_glove(params, batch):
        batch = [sent if sent != [] else ['.'] for sent in batch]
        embeddings = []

        for sent in batch:
            sentvec = []
            for word in sent:
                if word in params.word_vec:
                    sentvec.append(params.word_vec[word])
            if not sentvec:
                vec = np.zeros(params.wvec_dim)
                sentvec.append(vec)
            # Average of word embeddings as sentence embedding
            sentvec = np.mean(sentvec, 0)
            embeddings.append(sentvec)

        # Create N x D matrix (N sentences, D dimension)
        embeddings = np.vstack(embeddings)
        return embeddings

    # Add SE to evaluators
    evaluators["GLOVE"] = senteval.engine.SE(
        params, batcher_glove, prepare_glove)

    # FastText sentence embeddings -----------------------------------------------------------------------------------------

    # Set prepare and batcher functions
    def prepare_fast(params, samples):
        _, _, params.word2id = create_dictionary(samples)
        params.word_vec, params.wvec_dim = get_wordvec(
            FAST_FILE, params.word2id)

    def batcher_fast(params, batch):
        batch = [sent if sent != [] else ['.'] for sent in batch]
        embeddings = []

        for sent in batch:
            sentvec = []
            for word in sent:
                if word in params.word_vec:
                    sentvec.append(params.word_vec[word])
            if not sentvec:
                vec = np.zeros(params.wvec_dim)
                sentvec.append(vec)
            # Average of word embeddings as sentence embedding
            sentvec = np.mean(sentvec, 0)
            embeddings.append(sentvec)

        # Create N x D matrix (N sentences, D dimension)
        embeddings = np.vstack(embeddings)
        return embeddings

    # Add SE to evaluators
    evaluators["FAST"] = senteval.engine.SE(
        params, batcher_fast, prepare_fast)

    # Word2vec sentence embeddings -----------------------------------------------------------------------------------------

    # Set prepare and batcher functions
    def prepare_word2vec(params, samples):
        _, _, params.word2id = create_dictionary(samples)
        # Loading word2vec pre-trained matrix
        word_vectors = models.KeyedVectors.load_word2vec_format(
            WORD2VEC_FILE, binary=True)
        word_vec = {}
        for word in params.word2id:
            try:
                word_vec[word] = word_vectors.word_vec(word)
            except KeyError:  # if word not in pre-trained corpus vocabulary, skip
                continue
        params.word_vec, params.wvec_dim = word_vec, list(word_vec.values())[
            0].shape[0]

    def batcher_word2vec(params, batch):
        batch = [sent if sent != [] else ['.'] for sent in batch]
        embeddings = []

        for sent in batch:
            sentvec = []
            for word in sent:
                if word in params.word_vec:
                    sentvec.append(params.word_vec[word])
            if not sentvec:
                vec = np.zeros(params.wvec_dim)
                sentvec.append(vec)
            # Average of word embeddings as sentence embedding
            sentvec = np.mean(sentvec, 0)
            embeddings.append(sentvec)

        # Create N x D matrix (N sentences, D dimension)
        embeddings = np.vstack(embeddings)
        return embeddings

    # Add SE to evaluators
    evaluators["WORD2VEC"] = senteval.engine.SE(
        params, batcher_word2vec, prepare_word2vec)

    # Baseline embeddings Arora et al., 2017 -------------------------------------------------------------------------------

    # Create word_freq
    def create_word_freq(sentences):
        """
        Common crawl word frequencies like in the paper
        """
        words_freq = {}
        for s in sentences:
            for word in s:
                words_freq[word] = word_frequency(word, 'en')

        return words_freq

    # Set prepare and batcher functions

    def prepare_baseline(params, samples):
        _, _, params.word2id = create_dictionary(samples)
        params.word_freq = create_word_freq(samples)
        params.word_vec, params.wvec_dim = get_wordvec(
            GLOVE_FILE, params.word2id)
        params.a = 1e-3  # default value. needs tuning

    def batcher_baseline(params, batch):
        batch = [sent if sent != [] else ['.'] for sent in batch]

        # Creating embedding matrix
        n_samples = len(batch)
        embeddings = np.zeros((n_samples, params.wvec_dim))

        # Populating the embedding matrix
        for i in range(n_samples):
            sent = batch[i]  # sentence
            num_words = len(sent)  # number of words in sent
            word_vectors = []  # word vectors of sentence
            word_weights = []  # word weights of sentence
            for word in sent:
                if word in params.word_vec:
                    # vector current word
                    word_vectors.append(params.word_vec[word])
                    # weight for current word
                    word_weights.append(
                        params.a / (params.a + params.word_freq[word]))
            # computing embedding of sentence
            embeddings[i, :] = np.array(word_weights).dot(
                np.array(word_vectors)) / num_words

        # compute PCA of the sentence embeddings
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(embeddings)

        # remove the projection on the 1st principal component
        pc = svd.components_
        embeddings = embeddings - embeddings.dot(pc.transpose()) * pc

        return embeddings

    # Add SE to evaluators
    evaluators["ARORAETAL2016"] = senteval.engine.SE(
        params, batcher_baseline, prepare_baseline)

    # Evaluate each model  -------------------------------------------------------------------------------
    results = {}
    for name, se in evaluators.items():
        logger.info(f'Evaluating {name} model...')
        if skip_word2vec and name == "WORD2VEC":
            logger.info(
                f'WORD2VEC model will be skipped due to memory constraints...')
            continue
        evaluation = se.eval(transfer_tasks)
        # Removing 'STSBenchmark', 'yhat' if exists
        if 'STSBenchmark' in evaluation.keys():
            evaluation['STSBenchmark'].pop('yhat')
        results[name] = evaluation
        break

    # Saving evaluation results -------------------------------------------------------------------------------
    logger.info(f'Saving results to {OUTPUT_FILE}...')
    with open(OUTPUT_FILE, mode="w") as file:
        file.write(json.dumps(results, cls=ComplexEncoder, indent=4))


if __name__ == "__main__":
    # Set up logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Defining Paths
    PATH_TO_DATA = os.path.join(
        PROJECT_ROOT, "src", "senteval", "data")
    GLOVE_FILE = os.path.join(
        PROJECT_ROOT, "data", "external", "glove.840B.300d.txt")
    FAST_FILE = os.path.join(
        PROJECT_ROOT, "data", "external", "crawl-300d-2M.vec")
    WORD2VEC_FILE = os.path.join(
        PROJECT_ROOT, "data", "external", "GoogleNews-vectors-negative300.bin.gz")
    TFIDF_FILE = os.path.join(
        PROJECT_ROOT, "models", "saved_models", "TfidfVectorizer.joblib")
    BOW_FILE = os.path.join(
        PROJECT_ROOT, "models", "saved_models", "CountVectorizer.joblib")
    OUTPUT_FILE = os.path.join(
        PROJECT_ROOT, "models", "senteval_results.json")

    # Available tasks
    # ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
    # 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
    # 'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    # 'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
    # 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

    # Evaluation tasks to apply
    transfer_tasks = ['STS12', 'STS13', 'STS14',
                      'STS15', 'STS16', 'STSBenchmark']

    # Set params for SentEval
    params = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5}
    params['classifier'] = {
        'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}

    # Skipping word2vec model due to memory constraints
    skip_word2vec = True

    main()
