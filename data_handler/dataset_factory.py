import data_handler.dataset as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name):
        if name == "CIFAR100":
            return data.CIFAR100()
        elif name == "CIFAR10":
            return data.CIFAR10()
        elif name == 'MNIST':
            return data.MNIST()
        elif name == "CIFAR100_for_Resnet":
            return data.CIFAR100_for_Resnet()
