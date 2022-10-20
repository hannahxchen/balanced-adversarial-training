import numpy as np
from textattack.search_methods import SearchMethod

class SearchOnce(SearchMethod):
    """Search method for negation attack. It generates all transformations at once and search only once. 
    Does not depend on query budget.
    """
    def perform_search(self, initial_result):
        best_result = initial_result

        transformations = self.get_transformations(
            initial_result.attacked_text, original_text=initial_result.attacked_text
        )
        if len(transformations) == 0:
            return best_result

        results, _ = self.get_goal_results(transformations)
        scores = np.array([r.score for r in results])
        best_result = results[scores.argmax()]

        return best_result

    @property
    def is_black_box(self):
        return True