import random
from nltk import pos_tag
from textattack.transformations import WordSwap

bad_word_list = []
with open("../datasets/short_bad_word_list.txt", "r") as f:
    for line in f:
        bad_word_list.append(line.strip())

bad_word_dict = {}
for word in bad_word_list:
    tag = pos_tag([word], tagset="universal")[0][1]
    if tag not in bad_word_dict:
        bad_word_dict[tag] = [word]
    else:
        bad_word_dict[tag].append(word)

class WordSwapToxic(WordSwap):
    """Replace a word with a toxic word"""

    def _get_replacement_words(self, word):
        """Return a list bad words."""
        tag = pos_tag([word], tagset="universal")[0][1]
        if tag in bad_word_dict:
            replacement_words = bad_word_dict[tag]
            if len(replacement_words) <= 50:
                return replacement_words
            else:
                return random.sample(bad_word_dict[tag], 50)
        else:
            return []
