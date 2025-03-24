import math
import numpy as np
from utils import sentenize, tokenize_sentence

class TextRankSummarizer:
    def __init__(self, damping=0.85, epsilon=1e-4, niter=100,
                 preprocessing_function=tokenize_sentence,
                 similarity_function=None):
        self.damping = damping
        self.epsilon = epsilon
        self.niter = niter
        self.preprocessing_function = preprocessing_function
        if similarity_function is None:
            self.similarity_function = self.default_similarity
        else:
            self.similarity_function = similarity_function

    def default_similarity(self, tokens1, tokens2):
        intersection_size = sum(tokens2.count(w) for w in tokens1)
        if intersection_size == 0:
            return 0.0
        norm = math.log(len(tokens1)) + math.log(len(tokens2))
        return intersection_size / norm

    def __call__(self, text, target_sentences_count):
        original_sentences = sentenize(text)
        sentences = [self.preprocessing_function(s) for s in original_sentences]

        graph = self._create_graph(sentences)
        norm_graph = self._norm_graph(graph)
        ranks = self._iterate(norm_graph)

        indices = list(range(len(sentences)))
        indices = [idx for _, idx in sorted(zip(ranks, indices), reverse=True)]
        indices = indices[:target_sentences_count]
        indices.sort()
        return " ".join([original_sentences[idx] for idx in indices])

    def _create_graph(self, sentences):
        sentences_count = len(sentences)
        graph = np.zeros((sentences_count, sentences_count))
        for i in range(sentences_count):
            for j in range(i, sentences_count):
                sim = self.similarity_function(sentences[i], sentences[j])
                graph[i, j] = sim
                graph[j, i] = sim
        return graph

    def _norm_graph(self, graph):
        norm = graph.sum(axis=1, keepdims=True)
        norm_graph = graph / (norm + 1e-7)
        return norm_graph

    def _iterate(self, matrix):
        sentences_count = len(matrix)
        p_vector = np.full((sentences_count,), 1.0 / sentences_count)
        random_transitions = np.full((sentences_count,), 1.0 / sentences_count)
        transposed_matrix = matrix.T
        for _ in range(self.niter):
            next_p = (1.0 - self.damping) * random_transitions + self.damping * np.dot(transposed_matrix, p_vector)
            if np.linalg.norm(next_p - p_vector) < self.epsilon:
                break
            p_vector = next_p
        return p_vector