import re
from textattack.transformations import Transformation
from textattack.shared import AttackedText

bad_word_list = []
with open("../datasets/final_bad_word_list.txt", "r") as f:
    for line in f:
        bad_word_list.append(line.strip())

class RemoveToxic(Transformation):
    """Remove toxic words"""
    def _get_transformations(self, current_text, _):
        new_text = current_text
        count = 0
        
        for i, word in enumerate(current_text.words):
            word = word.lower()
            for bad_word in bad_word_list:
                if word == bad_word or bad_word in word:
                    new_text = new_text.delete_word_at_index(i-count)
                    count += 1
                    break

        if count == current_text.num_words: # skip examples with no words left
            return []
        if count > 0:
            return [new_text]
        else:
            return []