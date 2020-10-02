import sys
import io
import numpy as np
import logging
from wordfreq import word_frequency
from sklearn.decomposition import PCA
from scripts.senteval_train import SentevalTrain
import gensim.downloader as api
import json

# Set PATHs
PATH_TO_SENTEVAL = './SentEval/'
PATH_TO_DATA = './SentEval/data'
PATH_TO_OUT = './outputs/results.json'
PATH_TO_GLOVE = './SentEval/embeddings/glove.840B.300d.txt'
PATH_TO_FAST = './SentEval/embeddings/crawl-300d-2M.vec'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_results(results, path, format="json"):
    if format == "json":
        with open(path, mode="w") as file:
            file.write(json.dumps(results, cls=ComplexEncoder, indent=1))


# Getting Trained Models -----------------------------------------------------------------------------------------------
VOCAB_FILE = "./outputs/vocabulary_bow.json"
TFIDF_FILE = "./outputs/tfidf_vect.joblib"
# data_path = "./data/bbc"
# sent_train = SentevalTrain()
# sent_train.load_data(data_path)
# sent_train.train()
# sent_train.dump(vocab_file, tfidf_file)
count_vect, tfidf_vect = SentevalTrain.load(VOCAB_FILE, TFIDF_FILE)

evaluators = {}  # where we save each SE (Sentence Evaluator) object


# BOW sentence embeddings ----------------------------------------------------------------------------------------------
# SentEval prepare and batcher
def prepare_bow(params, samples):
    params.word2id = count_vect.vocabulary
    params.wvec_dim = len(params.word2id)


def batcher_bow(params, batch):
    batch = [" ".join(text) for text in batch]
    prep_batch = count_vect.transform(batch).toarray()
    return prep_batch


# Set params for SentEval
params_senteval_bow = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval_bow['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}

# Add SE to evaluators
evaluators["BOW"] = senteval.engine.SE(params_senteval_bow, batcher_bow, prepare_bow)


# TF-IDF sentence embeddings -------------------------------------------------------------------------------------------
# SentEval prepare and batcher
def prepare_tfidf(params, samples):
    params.word2id = tfidf_vect.named_steps["count"].vocabulary
    params.wvec_dim = len(params.word2id)


def batcher_tfidf(params, batch):
    batch = [" ".join(text) for text in batch]
    prep_batch = tfidf_vect.transform(batch).toarray()
    return prep_batch


# Set params for SentEval
params_senteval_tfidf = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval_tfidf['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}

# Add SE to evaluators
evaluators["TFIDF"] = senteval.engine.SE(params_senteval_tfidf, batcher_tfidf, prepare_tfidf)


# Glove sentence embeddings --------------------------------------------------------------------------------------------
# Create dictionary
def create_dictionary(sentences, threshold=0):
    words_count = {}
    for s in sentences:
        for word in s:
            words_count[word] = words_count.get(word, 0) + 1

    if threshold > 0:
        newwords_count = {}
        for word in words_count:
            if words_count[word] >= threshold:
                newwords_count[word] = words_count[word]
        words_count = newwords_count

    # Getting word frequencies
    total = sum(words_count.values())
    words_freq = {k: v / total for k, v in words_count.items()}

    words_freq['<s>'] = 1e9 + 4
    words_freq['</s>'] = 1e9 + 3
    words_freq['<p>'] = 1e9 + 2

    sorted_words = sorted(words_freq.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return words_freq, id2word, word2id


# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare_glove(params, samples):
    _, _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_GLOVE, params.word2id)
    params.wvec_dim = 300
    return


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
        sentvec = np.mean(sentvec, 0)  # Average of word embeddings as sentence embedding
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval_glove = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval_glove['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}

# Add SE to evaluators
evaluators["GLOVE"] = senteval.engine.SE(params_senteval_glove, batcher_glove, prepare_glove)


# FastText sentence embeddings -----------------------------------------------------------------------------------------
# SentEval prepare and batcher
def prepare_fast(params, samples):
    _, _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_GLOVE, params.word2id)
    params.wvec_dim = 300
    return


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
        sentvec = np.mean(sentvec, 0)  # Average of word embeddings as sentence embedding
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval_fast = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval_fast['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}

# Add SE to evaluators
evaluators["FAST"] = senteval.engine.SE(params_senteval_fast, batcher_fast, prepare_fast)


# Word2vec sentence embeddings -----------------------------------------------------------------------------------------
wv = api.load('word2vec-google-news-300')

def prepare_word2vec(params, samples):
    _, _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_GLOVE, params.word2id)
    params.wvec_dim = 300
    return


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
        sentvec = np.mean(sentvec, 0)  # Average of word embeddings as sentence embedding
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval_fast = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval_fast['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}

# Add SE to evaluators
evaluators["WORD2VEC"] = senteval.engine.SE(params_senteval_fast, batcher_fast, prepare_fast)


# Baseline embeddings Arora et al., 2017 -------------------------------------------------------------------------------
# Create word_freq
def create_word_freq(sentences):
    """
    Such as in the paper they used common crawl frequencies
    """
    words_freq = {}
    for s in sentences:
        for word in s:
            words_freq[word] = word_frequency(word, 'en')

    return words_freq


# SentEval prepare and batcher
def prepare_baseline(params, samples):
    _, _, params.word2id = create_dictionary(samples)
    params.word_freq = create_word_freq(samples)
    params.word_vec = get_wordvec(PATH_TO_GLOVE, params.word2id)
    params.wvec_dim = 300
    params.a = 1e-3  # default value. needs tuning
    return


def batcher_baseline(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []  # set of sentence embeddings

    for sent in batch:
        sentvec = np.zeros(params.wvec_dim)
        sentence_length = len(sent)
        for word in sent:
            if word in params.word_vec:
                sif = params.a / (params.a + params.word_freq[word])  # smooth inverse frequency, SIF
                sentvec = np.add(sentvec, np.multiply(sif, params.word_vec[word]))  # sentvec += sif * word_vector

        sentvec = np.divide(sentvec, sentence_length)  # weighted average
        embeddings.append(sentvec)  # add to our existing re-calculated set of sentences

    # calculate PCA of the sentence embeddings
    pca = PCA()
    pca.fit(np.array(embeddings))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < params.wvec_dim:
        for i in range(params.wvec_dim - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, sentvec = sentvec - u x uT x sentvec
    sentence_vecs = []
    for sentv in embeddings:
        sub = np.multiply(u, sentv)  # u x uT x sentvec
        sentence_vecs.append(np.subtract(sentv, sub))
    sentence_vecs = np.vstack(sentence_vecs)

    return sentence_vecs


# Set params for SentEval
params_senteval_baseline = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval_baseline['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3,
                                          'epoch_size': 2}

# Add SE to evaluators
evaluators["ARORAETAL2016"] = senteval.engine.SE(params_senteval_baseline, batcher_baseline, prepare_baseline)


if __name__ == "__main__":
    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark']
    results = {}
    # for name, se in evaluators.items():
    #     results[name] = se.eval(transfer_tasks)
    results["ARORAETAL2016"] = evaluators["ARORAETAL2016"].eval(transfer_tasks)
    # print("Saving results in {}".format(PATH_TO_OUT))
    # save_results(results, PATH_TO_OUT)
