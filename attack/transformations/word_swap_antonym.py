from nltk.corpus import wordnet
from textattack.transformations import WordSwap
from textattack.shared.utils import is_one_word

class WordSwapAntonym(WordSwap):
    """Transforms an input by replacing its words with antonyms provided by
    WordNet."""

    def __init__(self, language="eng"):
        if language not in wordnet.langs():
            raise ValueError(f"Language {language} not one of {wordnet.langs()}")
        self.language = language

    def _get_replacement_words(self, word, random=False):
        """Return a list containing all antonyms found for the target word from WordNet."""
        antonyms = set()
        for synset in wordnet.synsets(word, lang=self.language):
            for lemma in synset.lemmas(lang=self.language):
                if lemma.antonyms():
                    antonym_word = lemma.antonyms()[0].name()
                    if (
                        ("_" not in antonym_word) 
                        and (is_one_word(antonym_word))
                    ):
                        antonyms.add(antonym_word)

        return antonyms

        