"""
This code is modified by Dmitrii from the original code by Nicholas Carlini <nicholas@carlini.com>.
"""

import numpy as np
import os
import gzip
import urllib.request


class MnistLoader:
    def __init__(self):
        self.VALIDATION_SIZE = 5000

        self.train_data = None
        self.train_labels = None

        self.validation_data = None
        self.validation_labels = None

        self.test_data = None
        self.test_labels = None

    def load_data(self, dataset_folder=None):
        data_path = os.path.join(dataset_folder, "mnist_data")
        if not os.path.exists(data_path):
            print("Downloading MNIST dataset online")
            self._download_mnist_data(data_path)
        else:
            print("Loading MNIST dataset from a local folder")

        train_data = self._extract_inputs(f"{data_path}/train-images-idx3-ubyte.gz", 60000)
        train_labels = self._extract_labels(f"{data_path}/train-labels-idx1-ubyte.gz", 60000)

        self.test_data = self._extract_inputs(f"{data_path}/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = self._extract_labels(f"{data_path}/t10k-labels-idx1-ubyte.gz", 10000)

        self.validation_data = train_data[:self.VALIDATION_SIZE, :]
        self.validation_labels = train_labels[:self.VALIDATION_SIZE]

        self.train_data = train_data[self.VALIDATION_SIZE:, :]
        self.train_labels = train_labels[self.VALIDATION_SIZE:]

    @staticmethod
    def _download_mnist_data(data_path):
        os.mkdir(data_path)
        files = ["train-images-idx3-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz",
                 "train-labels-idx1-ubyte.gz",
                 "t10k-labels-idx1-ubyte.gz"]
        for name in files:
            urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, f"{data_path}/" + name)

    @staticmethod
    def _extract_inputs(filename, num_images):
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(num_images * 28 * 28)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            # todo note this way of normalisation: see MNISTAttackedModel.predict for details
            # data = (data / 255) - 0.5
            data = (data / 255)
            data = data.reshape(num_images, -1)
            return data

    @staticmethod
    def _extract_labels(filename, num_images):
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8)
        num_classes = 10
        return (np.arange(num_classes) == labels[:, None]).astype(np.float32)
