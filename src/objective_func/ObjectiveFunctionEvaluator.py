"""
A class for evaluating the objective function of the attack
"""
from attacked_models.AttackedModelFactory import AttackedModelFactory
import numpy as np


class ObjectiveFunctionEvaluator:
    def __init__(self, dataset_descriptor, target_image, use_softmax, target_class=None):
        """

        :type use_softmax: bool. A flag whether to use sofmax for the output of the attacked model
        :type target_class: int. A target class for the attack.
        :type target_image: the target image for the attack.
        :type dataset_descriptor: the descriptor of the attack dataset.
        """
        self.use_softmax = use_softmax
        self.target_class = target_class
        self.dataset_descriptor = dataset_descriptor
        self.target_image = target_image
        self.attacked_model = AttackedModelFactory().get_attacked_model(dataset_descriptor=self.dataset_descriptor,
                                                                        use_softmax=self.use_softmax)
        # the correct class of the original target image we are trying to mis-classify
        self.correct_class = np.argmax(self.attacked_model.predict(self.target_image))

    # We want to maximize the objective function to misclassify the image.
    # If the objective function is larger than zero, the attack is successful
    def evaluate(self, perturbation):
        perturbed_image = self.target_image + self.dataset_descriptor.epsilon * perturbation
        perturbed_image_predictions = self.attacked_model.predict(perturbed_image)
        # the score of the correct class of the original target image
        correct_class_score = perturbed_image_predictions[self.correct_class]

        # first implement the case with probabilities as outputs
        if self.use_softmax:
            # the case of the targeted attack
            if self.target_class:
                return perturbed_image_predictions[self.target_class] - correct_class_score
            # the case of the untargeted attack
            else:
                perturbed_image_predictions[self.correct_class] = -float('inf')
                # the best score of all incorrect classes. We want it to be larger than the score of the correct class.
                best_incorrect_class_score = np.max(perturbed_image_predictions)
                return best_incorrect_class_score - correct_class_score
        else:
            raise NotImplementedError
