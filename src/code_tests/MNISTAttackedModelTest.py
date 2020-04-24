import numpy as np

from attacked_models.AttackedModelFactory import AttackedModelFactory
from dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from dataset_data.mnist.MnistLoader import MnistLoader

if __name__ == '__main__':
    latent_dim = 10
    mnist_root = '../../datasets/'

    dataset_descriptor = MNISTDescriptor()
    # vae = VAEFactory.get_vae(dataset_descriptor=dataset_descriptor,
    #                          latent_dimension=latent_dim)
    # vae.train(num_epochs=50, dataset_folder=mnist_root)

    # x = np.atleast_2d(np.ones(latent_dim))
    # y = vae.decode(x)
    # print(y.shape)

    mnist_loader = MnistLoader()
    dataset_descriptor.dataset_folder = '../../datasets/'
    mnist_loader.load_data(dataset_folder=dataset_descriptor.dataset_folder)
    # print(mnist_loader.test_data.shape)

    test_images = mnist_loader.test_data

    dataset_descriptor.attacked_model_path = '../attacked_models/mnist/mnist'
    attacked_model = AttackedModelFactory.get_attacked_model(dataset_descriptor)

    predictions = attacked_model.predict(test_images)

    predcited_classes = predictions.argmax(axis=1)
    correct_classes = mnist_loader.test_labels[:, :].argmax(axis=1)
    # print(mnist_loader.test_labels[:, :].argmax(axis=1))
    print("accuracy is {}".format(np.mean(correct_classes == predcited_classes)))