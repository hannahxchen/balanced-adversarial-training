from textattack import Attack
from textattack.search_methods import GreedyWordSwapWIR
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import InputColumnModification, StopwordModification

from goal_functions import RemainPrediction
from transformations import WordSwapAntonym

from textattack.attack_recipes import AttackRecipe

class AntonymAttack(AttackRecipe):
    """Antonym attack"""

    @staticmethod
    def build(model_wrapper, skip_column=None, skip_label=None):
        transformation = WordSwapAntonym()

        constraints = [
            StopwordModification(), 
            PartOfSpeech(allow_verb_noun_swap=False),
        ]
        constraints.append(MaxWordsPerturbed(max_num_words=1))

        if skip_column:
            input_column_modification = InputColumnModification(
                ["premise", "hypothesis"], {skip_column}
            )
            constraints.append(input_column_modification)
       
        goal_function = RemainPrediction(model_wrapper, skip_label=skip_label)
        search_method = GreedyWordSwapWIR(wir_method="gradient")

        return Attack(goal_function, constraints, transformation, search_method)
