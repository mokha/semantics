from semeval.common import download_path
import hickle as hkl
import numpy as np
import io
from sklearn import preprocessing
import pathlib


class Relatedness:
    def __init__(self, lang='eng'):

        rows_path = "{}{}-relatedness-{}.{}".format(download_path(), lang, 'rows', 'txt')
        cols_path = "{}{}-relatedness-{}.{}".format(download_path(), lang, 'cols', 'txt')
        matrix_path = "{}{}-relatedness-{}.{}".format(download_path(), lang, 'model', 'hkl')

        if not pathlib.Path(rows_path).is_file() or \
                not pathlib.Path(cols_path).is_file() or \
                not pathlib.Path(matrix_path).is_file():
            raise Exception(
                "Relatedness models are not download. Download them using `python -m semeval.download -m relatedness -l {}`"
                    .format(lang)
            )

        # get the rows and their indecies
        self.rows = Relatedness.load_termidx(rows_path)
        # get the columns and their indecies
        self.cols = Relatedness.load_termidx(cols_path)
        # load the matrix
        self.matrix = hkl.load(matrix_path)

        self.rev_cols = {i: r for r, i in self.cols.items()}
        sorted_cols = sorted(self.cols.items(), key=lambda k: k[1])
        self.sorted_cols = [r[0] for r in sorted_cols]

    @staticmethod
    def load_termidx(filename):
        with io.open(filename, 'r', encoding='utf-8') as f:
            rows = f.readlines()
            rows = [r.rstrip('\n').split('\t')[0] for r in rows]
            return dict([(w, i) for i, w in enumerate(rows)])

    def get_vector(self, word, normalize=True):
        try:
            i = self.rows[word]  # get the index of the word in the matrix
            row = self.matrix[i, :].A[0]  # get it's relatedness vector/row
            if normalize:
                row = preprocessing.normalize(row[:, np.newaxis], axis=0, norm='l1').ravel()
            return row
        except Exception as e:
            return None

    def get_rel(self, word, normalize=True, positive=True):
        try:
            i = self.rows[word]  # get the index of the word in the matrix
            row = self.matrix[i, :].A[0]  # get it's relatedness vector/row
            if normalize:
                row = preprocessing.normalize(row[:, np.newaxis], axis=0, norm='l1').ravel()
            row = zip(self.sorted_cols, row)  # word: relatedness_score
            if positive:
                row = [r for r in row if r[1] > 0]  # remove non-related words
            row = dict(row)
            return row
        except Exception as e:
            return None

    def get_sorted_rel(self, word, normalize=True, positive=True, k=0):
        row = self.get_rel(word, normalize, positive)
        if row:
            row = sorted(row.items(), key=lambda k: k[1], reverse=True)
            if k > 0 and k < len(row):
                row = row[:k]
        return row

    def interpret(self, tenor, vehicle):
        """
        The "Combined rank" interpretation method as presented in
        Xiao, P., Alnajjar, K., Granroth-Wilding, M., Agres, K., & Toivonen, H. (2016). Meta4meaning: Automatic Metaphor
         Interpretation Using Corpus-Derived Word Associations. In Proceedings of The Seventh International Conference
         on Computational Creativity (pp. 230-237). Sony CSL Paris.
        """
        tv = self.get_vector(tenor)
        vv = self.get_vector(vehicle)

        if tv is None or vv is None:
            return []

        features = np.union1d(np.where(tv > 0), np.where(vv > 0))  # all non-zero features
        shared_features = np.intersect1d(np.where(tv > 0), np.where(vv > 0))  # shared features

        # consider only concrete features? add filter here (e.g., certain POS tags)

        mv = np.zeros(tv.shape)
        mv[shared_features] = tv[shared_features] * vv[shared_features]  # multiplication
        mv = mv[features]

        odv = np.zeros(tv.shape)
        odv[features] = float('-inf')
        odv[shared_features] = vv[shared_features] - tv[shared_features]  # overlap difference
        odv = odv[features]

        # sort them and convert back into ordered features
        mvf = np.argsort(-mv)
        odvf = np.argsort(-odv)

        mvf_l = mvf.tolist()
        odvf_l = odvf.tolist()

        # ranked interpretations
        interpretations = [(self.rev_cols[features[f]], min([mvf_l.index(f), odvf_l.index(f)])) for f in
                           range(len(features))]
        return list(sorted(interpretations, key=lambda k: k[1]))

    def metaphoricity(self, tenor, vehicle, expression, k=0, normalize=True):
        """
        Metaphoricity scores as described in Alnajjar, K., & Toivonen, H. (2020). Computational Generation of Slogans.
         Natural Language Engineering. https://doi.org/10.1017/S1351324920000236
        """
        tv = self.get_sorted_rel(tenor, normalize=normalize)
        vv = self.get_sorted_rel(vehicle, normalize=normalize)

        if not tv or not vv:
            return 0.0

        if k == 0:
            pass
        elif k > 0 and k < len(self.rows):
            tv = tv[:k]
            vv = vv[:k]
        else:
            raise Exception("k must be positive and less than %s" % len(self.rows))

        tv = dict(tv)
        vv = dict(vv)

        t_relatedness = np.max([tv.get(t, 0.0) for t in expression])  # relatedness to tenor
        v_relatedness = np.max([vv.get(t, 0.0) for t in expression])  # relatedness to vehicle
        tv_score = t_relatedness * v_relatedness

        vt_diff = np.max([vv.get(t, 0.0) - tv.get(t, 0.0) for t in expression])  # vehicle tenor difference

        return tv_score, vt_diff, np.mean([tv_score, vt_diff]) if tv_score > 0 and vt_diff > 0 else 0.0


def main():
    from pprint import pprint

    relatedness_m = Relatedness(lang='eng')

    pprint(relatedness_m.get_sorted_rel('car')[:5])
    print()

    pprint(relatedness_m.interpret('alcohol', 'crutch')[:10])
    print()
    pprint(relatedness_m.interpret('cloud', 'cotton')[:10])
    print()

    # 0 to select all
    pprint(relatedness_m.metaphoricity('life', 'journey', ['I', 'walked', 'in', 'his', 'sorrows', '.'], 300))


if __name__ == '__main__':
    main()
