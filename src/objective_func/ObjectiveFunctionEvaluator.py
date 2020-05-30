"""
A class for evaluating the objective function of the attack
"""
from src.attacked_models.AttackedModelFactory import AttackedModelFactory
import numpy as np


class ObjectiveFunctionEvaluator:
    def __init__(self, dataset_descriptor, target_image, use_softmax, rescale, target_class=None):
        """

        :type rescale: bool. Rescale the image using epsilon value of just clip it to min/max values.
        :type use_softmax: bool. A flag whether to use sofmax for the output of the attacked model
        :type target_class: int. A target class for the attack.
        :type target_image: the target image for the attack. Pixels normalised to be in [0,1] range
        :type dataset_descriptor: the descriptor of the attack dataset.
        """
        self.rescale = rescale
        self.use_softmax = use_softmax
        self.target_class = target_class
        self.dataset_descriptor = dataset_descriptor
        self.target_image = target_image
        self.attacked_model = AttackedModelFactory().get_attacked_model(dataset_descriptor=self.dataset_descriptor,
                                                                        use_softmax=self.use_softmax)
        # the correct class of the original target image we are trying to mis-classify
        self.correct_class = np.argmax(self.attacked_model.predict(self.target_image))
        assert self.correct_class != self.target_class, "the target class should be different from the correct one"

    # perturbation is the image from the VAE, its pixels are normlised to be in [0,1] range
    def _get_perturbed_image(self, perturbation):

        assert perturbation.shape == self.target_image.shape, \
            "target and perturbation have wrong shapes {} and {}".format(self.target_image.shape, perturbation.shape)

        epsilon = self.dataset_descriptor.epsilon

        # rescale both images to the range [-0.5, 0.5]
        shifted_target_image = self.target_image - 0.5
        sifted_perturbation = perturbation - 0.5

        # unscaled_perturbed_image is in the range [-0.5 - eps/2, 0.5 + eps/2]
        unscaled_perturbed_image = shifted_target_image + epsilon * sifted_perturbation
        if self.rescale:
            perturbed_image = unscaled_perturbed_image  / (1 + epsilon)
        else:
            perturbed_image = unscaled_perturbed_image.clip(-0.5, 0.5)

        return perturbed_image

    # We want to minimize the objective function to misclassify the image.
    # If the objective function is less than zero, the attack is successful
    # perturbation is the image from the VAE, its pixels are normlised to be in [0,1] range
    def evaluate(self, perturbation):
        # perturbed_image is in the interval [-0.5, 0.5]
        perturbed_image = self._get_perturbed_image(perturbation=perturbation.squeeze())
        #  make the predictions 1-D
        perturbed_image_predictions = self.attacked_model.predict(perturbed_image).squeeze()

        # first implement the case with probabilities as outputs
        if self.use_softmax:
            # the case of the targeted attack
            if self.target_class is not None:
                # the score of the target class for the attack
                target_class_score = perturbed_image_predictions[self.target_class]

                # the best score of all non-target classes. We want it to be smaller than the score of the target class.
                perturbed_image_predictions[self.target_class] = -float('inf')
                best_other_class_score = np.max(perturbed_image_predictions)

                return  best_other_class_score - target_class_score
            # the case of the un-targeted attack
            else:
                # the score of the correct class of the original target image
                correct_class_score = perturbed_image_predictions[self.correct_class]

                # the best score of all incorrect classes. We want it to be larger than the score of the correct class.
                perturbed_image_predictions[self.correct_class] = -float('inf')
                best_incorrect_class_score = np.max(perturbed_image_predictions)

                return  correct_class_score - best_incorrect_class_score
        else:
            raise NotImplementedError
