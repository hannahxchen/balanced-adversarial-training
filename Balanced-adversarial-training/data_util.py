import json
import random
import numpy as np
import torch
import string

class SynonymWordSubstitude:
    def __init__(self, table):
        self.table = table
        self.table_key = set(list(table.keys()))
        self.exclude = set(string.punctuation)

    def get_perturbed_sentence(self, text):
        tem_text = [t for t in text.split(' ') if t != '']
        if tem_text[0]:
            for j in range(len(tem_text)):
                word = tem_text[j].lower()
                if word[-1] in self.exclude:
                    tem_text[j] = self.sample_from_table(word[0:-1]) + word[-1]
                else:
                    tem_text[j] = self.sample_from_table(word)
            perturbed_text = ' '.join(tem_text)
            return perturbed_text

    def sample_from_table(self, word):
        if word in self.table_key:
            tem_words = self.table[word]['set']
            num_words = len(tem_words)
            index = np.random.randint(0, num_words)
            return tem_words[index]
        else:
            return word


class AntonymWordSubstitude:
    def __init__(self, table):
        self.table = table
        self.table_key = set(list(table.keys()))
        self.exclude = set(string.punctuation)

    def get_perturbed_sentence(self, text):
        tem_text = [t for t in text.split(' ') if t != '']
        replacement_order = list(range(len(tem_text)))
        random.shuffle(replacement_order)

        if tem_text[0]:
            for i in replacement_order:
                word = tem_text[i].lower()
                if word[-1] in self.exclude:
                    perturb_word = self.sample_from_table(word[0:-1])
                else:
                    perturb_word = self.sample_from_table(word)

                if perturb_word is None:
                    continue
                else:
                    if word[-1] in self.exclude:
                        perturb_word += word[-1]
                    tem_text[i] = perturb_word
                    return " ".join(tem_text)

        return text

    def sample_from_table(self, word):
        if word in self.table_key:
            tem_words = self.table[word]['set']
            num_words = len(tem_words)
            index = np.random.randint(0, num_words)
            return tem_words[index]
        else:
            return None