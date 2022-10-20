from textattack.constraints import PreTransformationConstraint
from textattack.shared.validators import transformation_consists_of

class SkipNegatedVerbs(PreTransformationConstraint):
    """Return non-negated verb indices."""
    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        modifiable_indices = set()
        num_words = current_text.num_words
        words = current_text.words
        for i in range(num_words):
            if current_text.pos_of_word_index(i) != "VERB":
                continue
            
            negated = False
            if i > 0 and (words[i-1].endswith("n't") or words[i-1] == "not"):
                negated = True
            elif i < num_words - 1 and (words[i+1].endswith("n't") or words[i+1] == "not"):
                negated = True
            
            if not negated:
                modifiable_indices.add(i)
                
        return modifiable_indices

    def check_compatibility(self, transformation):
        """This constraint is only for AddNot transformation.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        from transformations import AddNot
        return transformation_consists_of(transformation, [AddNot])