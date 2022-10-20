import pickle
from textattack.transformations import WordSwap

class WordSwapSAFER(WordSwap):
    """Word substitution method from SAFER: 
    GLOVE embedding with counter-fitting & all-but-the-top post-processing
    """
    def __init__(self, word_sub_table_file):
        self.word_sub_table_file = word_sub_table_file

    def _get_replacement_words(self, word):
        """Returns a list of replacement words from the given word substitute table"""
        with open(self.word_sub_table_file, "rb") as f:
            table = pickle.load(f)

        if word in table:
            perturbation_set = table[word]["set"]
            perturbation_set.remove(word)
        else:
            perturbation_set = []
        return perturbation_set