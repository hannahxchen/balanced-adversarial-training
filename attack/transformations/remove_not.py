from textattack.transformations import Transformation

class RemoveNot(Transformation):
    """Remove not and no"""
    def _get_transformations(self, current_text, _):
        transformed_texts = []
        
        for i, word in enumerate(current_text.words):
            word = word.lower()
            new_text = None
            
            if word == "no":
                new_text = current_text.delete_word_at_index(i)
            elif word == "not":
                if i < current_text.num_words - 1 and current_text.words[i+1] == "only": # exclude "not only"
                    continue
                new_text = current_text.delete_word_at_index(i)
            elif word.endswith("n't"):
                if word == "can't":
                    new_text = current_text.replace_word_at_index(i, "can")
                elif word == "won't":
                    new_text = current_text.replace_word_at_index(i, "will")
                else:
                    new_text = current_text.replace_word_at_index(i, word.replace("n't", ""))

            if new_text is not None:
                transformed_texts.append(new_text)
                
        return transformed_texts