"""
A class for a wrapper for VAE
"""


class AbstractVariationalAutoEncoderWrapper:

    def __init__(self, dimension):
        """
        :type dimension: int. The dimension of VAE
        """
        pass

    # train the VAE
    def train(self, num_epochs=50):
        pass

    # encode the range of inputs - used to encode the history of BO inputs
    def encode_range(self, inputs):
        return []

    # decode the range of inputs - used to encode the history of BO inputs
    def decode_range(self, inputs):
        return []

    # encode one input
    def encode(self, input):
        return 0

    # decode one input
    def decode(self, input):
        return 0
