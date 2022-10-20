from flair.data import Sentence
from flair.models import SequenceTagger
from nltk.stem.wordnet import WordNetLemmatizer
from textattack.transformations import Transformation
from textattack.shared.utils import words_from_text

contracted_verbs = [
    "i'm", "you're", "they're", "he's", "she's", "it's",
    "there's", "that's", "how's", "what's", "when's", "where's", "who's"
    "i've", "you've", "they've", "must've", "should've", "would've", "might've",
    "i'd", "you'd", "they'd", "he'd", "she'd", "it'd", "where'd", "how'd", "there'd", "who'd",
    "i'll", "you'll", "they'll", "he'll", "she'll", "it'll", "how'll", "who'll"
]

def zip_flair_result(pred):
    """Takes a sentence tagging from `flair` and returns two lists, of words
    and their corresponding parts-of-speech."""

    tokens = pred.tokens
    word_list = []
    pos_list = []
    for token in tokens:
        word_list.append(token.text)
        pos_list.append(token.annotation_layers["pos"][0]._value)

    return word_list, pos_list

class AddNot(Transformation):
    """Add not before/after a verb"""
    def __init__(self):
        self.pos_tagger = SequenceTagger.load('pos-fast')
        self.lemmatizer = WordNetLemmatizer()

    def _get_transformations(self, current_text, indices_to_modify):
        sentence = Sentence(current_text.text, use_tokenizer=words_from_text)
        self.pos_tagger.predict(sentence)
        word_list, pos_list = zip_flair_result(sentence)
        transformed_texts = []
        
        for i in indices_to_modify:
            word, pos = word_list[i].lower(), pos_list[i]
            new_text = None
            
            if word in contracted_verbs: # add negation for contracted Be verbs
                new_text = current_text.insert_text_after_word_index(i, "not")

            elif pos in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                if pos in ["VB", "VBG", "VBN"]: # verb: base form, gerund or present particple, past participle
                    new_text = current_text.insert_text_before_word_index(i, "not")
                    
                elif pos == "VBD": # verb, past tense
                    if word in ["was", "were", "did"]:
                        new_text = current_text.replace_word_at_index(i, word + "n't")
                    else:
                        present = self.lemmatizer.lemmatize(word, "v")
                        if present != word:
                            new_text = current_text.replace_word_at_index(i, present)
                            new_text = new_text.insert_text_before_word_index(i, "didn't")

                elif pos == "VBP": # verb, non-3rd person singular present
                    if word == "am":
                        new_text = current_text.insert_text_after_word_index(i, "not")
                    elif word in ["are", "do"]:
                        new_text = current_text.replace_word_at_index(i, word + "n't")
                    else:
                        new_text = current_text.insert_text_before_word_index(i, "don't")
                elif pos == "VBZ": # verb, 3rd person singular present
                    if word == "is":
                        new_text = current_text.replace_word_at_index(i, word + "n't")
                    else:
                        present = self.lemmatizer.lemmatize(word, "v")
                        if present and present != word:
                            new_text = current_text.replace_word_at_index(i, present)
                            new_text = new_text.insert_text_before_word_index(i, "doesn't")

            if new_text is not None:
                transformed_texts.append(new_text)
                
        return transformed_texts