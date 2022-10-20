from textattack import Attack
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation import InputColumnModification

from goal_functions import RemainPrediction
from transformations import AddNot, RemoveNot
from constraints import SkipNegatedVerbs
from search_methods import SearchOnce

from textattack.attack_recipes import AttackRecipe

class NegationAttack(AttackRecipe):
    """Negation attack
    """

    @staticmethod
    def build(model_wrapper, skip_column=None, skip_label=None):

        transformation = CompositeTransformation([AddNot(), RemoveNot()])

        constraints = [SkipNegatedVerbs()]
        constraints.append(MaxWordsPerturbed(max_num_words=1))

        if skip_column:
            input_column_modification = InputColumnModification(
                ["premise", "hypothesis"], {skip_column}
            )
            constraints.append(input_column_modification)
       
        goal_function = RemainPrediction(model_wrapper, skip_label=skip_label)
        search_method = SearchOnce()

        return Attack(goal_function, constraints, transformation, search_method)