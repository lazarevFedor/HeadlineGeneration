import math
import numpy as np
from utils import tokenize_sentence

class LexRankSummarizer:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self._idf = {}

    def __call__(self, sentences, sentence_count=1):
        if not sentences:
            return []

        processed = []
        doc_freq = {}
        for i, s in enumerate(sentences):
            tokens = tokenize_sentence(s)
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
            processed.append({'index': i, 'text': s, 'tf': tf})
        total_docs = len(processed)
        for sent in processed:
            norm_sq = 0.0
            for token, count in sent['tf'].items():
                idf = math.log(total_docs / (1 + doc_freq[token]))
                self._idf[token] = idf
                norm_sq += (count * idf) ** 2
            sent['norm'] = math.sqrt(norm_sq)

        matrix = self._create_similarity_matrix(processed)
        markov_matrix = self._normalize_matrix(matrix)
        ranks = self._power_method(markov_matrix)

        indexes = self._argsort(ranks, reverse=True)
        selected = sorted(indexes[:sentence_count])
        return [sentences[i] for i in selected]

    def _create_similarity_matrix(self, processed):
        n = len(processed)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                matrix[i][j] = self._get_similarity(processed[i], processed[j])
        return matrix

    def _get_similarity(self, sent1, sent2):
        if not sent1['tf'] or not sent2['tf']:
            return 0.0
        numerator = 0.0
        common = set(sent1['tf'].keys()) & set(sent2['tf'].keys())
        for token in common:
            tf1 = sent1['tf'][token]
            tf2 = sent2['tf'][token]
            idf = self._idf.get(token, 0.0)
            numerator += (tf1 * idf) * (tf2 * idf)
        denominator = sent1['norm'] * sent2['norm']
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    def _normalize_matrix(self, matrix):
        n = matrix.shape[0]
        for i in range(n):
            row_sum = matrix[i].sum()
            if row_sum != 0:
                matrix[i] = matrix[i] / row_sum
        return matrix

    def _power_method(self, matrix, epsilon=1e-6, max_iterations=100):
        n = matrix.shape[0]
        p = np.array([1.0 / n] * n)
        for _ in range(max_iterations):
            new_p = matrix.T.dot(p)
            if np.abs(new_p - p).sum() < epsilon:
                return new_p
            p = new_p
        return p

    def _argsort(self, seq, reverse=False):
        return sorted(range(len(seq)), key=lambda i: seq[i], reverse=reverse)