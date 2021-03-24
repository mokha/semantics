from gensim.models import KeyedVectors
import numpy as np
from pathlib import Path
from .common import *


class Embeddings(object):
    def __init__(self, lang='eng'):
        model_path = Path(download_path() + "vectors-{}.txt".format(lang))
        if lang not in supported_languages():
            raise Exception("Language '{}' is not supported!".format(lang))
        elif not model_path.is_file():
            raise Exception("Vectors for language '{}' are not downloaded! "
                            "Download them using `python -m semeval.download -m {} -l {}`"
                            .format(lang, 'embeddings', lang))

        self.L = KeyedVectors.load_word2vec_format(model_path, binary=False, unicode_errors='replace')
        self.L_len = len(self.L.vocab)

    @staticmethod
    def align(e1, e2, word, topn=10):
        return e2.L.similar_by_vector(e1.L.word_vec(word), topn=topn)

    def vector(self, word):
        return self.L[word]

    def project(self, word, e2, topn=10):
        return self.align(self, e2, word, topn=topn)

    def similarity(self, *args, **kwargs):
        return self.L.similarity(*args, **kwargs)

    def most_similar(self, *args, **kwargs):
        return self.L.most_similar_cosmul(*args, **kwargs)

    def theme(self, l):
        return self.L.most_similar(positive=l, topn=1)[0]

    def vocabulary(self):
        return self.L.vocab

    def centroid(self, l):
        vectors = [self.L.get_vector(w) for w in l if w in self.L.vocab]
        return np.mean(vectors, axis=0)

    def neighbours(self, w, topn=50):
        return self.L.most_similar(w, topn=topn)

    def neighbours_threshold(self, w, threshold=0.8):
        return [x for x in self.L.most_similar(w, topn=self.L_len) if x[1] >= threshold]

    def analogy(self, a, b, c, topn=10):
        return list(self.L.most_similar(positive=[b, c], negative=[a], topn=topn))

    def to_vector(self, tokens):
        text_v = [self.L.get_vector(t) for t in tokens if t in self.L.vocab]

        if len(text_v) == 0:
            raise Exception("No words in the text found in the model.")
        return np.mean(text_v, axis=0)
