from textattack.goal_functions import ClassificationGoalFunction

class RemainPrediction(ClassificationGoalFunction):
    """Attack for maintaining the same predicted label.
    
    Args:
        target_min_score (float): If set, goal is to remain the prediction above this score
        skip_label (int): label index that should be skipped
    """

    def __init__(self, *args, target_min_score=None, skip_label=None, **kwargs):
        self.target_min_score = target_min_score
        self.skip_label = skip_label
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, attacked_text):
        """check if attacked_text is different from the original text and 
        if the predicted class remains the smae
        """
        if self.initial_attacked_text.words == attacked_text.words:
            return False
        if self.target_min_score:
            return model_output[self.ground_truth_output] > self.target_min_score
        else:
            return model_output.argmax() == self.ground_truth_output

    def _should_skip(self, model_output, attacked_text):
        if self.skip_label is not None and self.ground_truth_output == self.skip_label:
            return True
        else:
            return self.ground_truth_output != model_output.argmax()
    
    def _get_score(self, model_output, _):
        # Give the lowest score possible to inputs which don't maintain the ground truth label.
        if self.ground_truth_output != model_output.argmax():
            return 0
        else:
            return model_output[self.ground_truth_output]

    def extra_repr_keys(self):
        return ["target_min_score", "skip_label"]
