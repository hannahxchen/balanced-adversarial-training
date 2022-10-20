from textattack.constraints import PreTransformationConstraint
from textattack.shared.validators import transformation_consists_of_word_swaps

class AllowedPoSTags(PreTransformationConstraint):
    
    def __init__(self, allowed_pos_tags=['VERB', "ADJ", "ADV", "NOUN"]):
        self.allowed_pos_tags = allowed_pos_tags

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        modifiable_indices = set()
        for i in range(current_text.num_words):
            if current_text.pos_of_word_index(i) in self.allowed_pos_tags:
                modifiable_indices.add(i)
        return modifiable_indices

    def check_compatibility(self, transformation):
        """The stopword constraint only is concerned with word swaps since
        paraphrasing phrases containing stopwords is OK.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return transformation_consists_of_word_swaps(transformation)