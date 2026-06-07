import numpy as np
from collections import defaultdict
from math import log


class CustomMultinomialNB:

    def __init__(self):
        self.spam_word_counts = defaultdict(int)
        self.ham_word_counts = defaultdict(int)

        self.spam_messages = 0
        self.ham_messages = 0

        self.spam_total_words = 0
        self.ham_total_words = 0

        self.vocabulary = set()

    def fit(self, texts, labels):

        for text, label in zip(texts, labels):

            words = text.split()

            if label == 1:
                self.spam_messages += 1

                for word in words:
                    self.spam_word_counts[word] += 1
                    self.spam_total_words += 1
                    self.vocabulary.add(word)

            else:
                self.ham_messages += 1

                for word in words:
                    self.ham_word_counts[word] += 1
                    self.ham_total_words += 1
                    self.vocabulary.add(word)

        self.total_messages = self.spam_messages + self.ham_messages

    def predict(self, texts):

        predictions = []

        vocab_size = len(self.vocabulary)

        spam_prior = log(self.spam_messages / self.total_messages)
        ham_prior = log(self.ham_messages / self.total_messages)

        for text in texts:

            words = text.split()

            spam_score = spam_prior
            ham_score = ham_prior

            for word in words:

                spam_word_probability = (
                    self.spam_word_counts[word] + 1
                ) / (self.spam_total_words + vocab_size)

                ham_word_probability = (
                    self.ham_word_counts[word] + 1
                ) / (self.ham_total_words + vocab_size)

                spam_score += log(spam_word_probability)
                ham_score += log(ham_word_probability)

            prediction = 1 if spam_score > ham_score else 0

            predictions.append(prediction)

        return np.array(predictions)