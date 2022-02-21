import torch
import networks.network as net
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model, task_info):
        if model == 'resnet18':
            return net.resnet18(task_info)
        elif model == 'MLP':
            return net.MLP(task_info)
