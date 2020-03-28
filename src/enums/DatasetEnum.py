def my_enum(**enums):
    return type('Enum', (), enums)


DatasetEnum = my_enum(MNIST=1,
                      CIFAR10=2)
