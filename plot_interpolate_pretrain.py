import sys, os, time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import pickle
import torch
from arguments import get_args
import random
import utils

import data_handler
from sklearn.utils import shuffle
import trainer
import networks
import copy
import torchvision

def compute_distance(model1, model2):
    norm_square_sum = 0
    for module1, module2 in zip(model1.modules(), model2.modules()):
        if 'Conv' in str(type(module1)):
            norm_square_sum += torch.norm(module1.weight.data-module2.weight.data)**2
            if module1.bias is not None: 
                norm_square_sum += torch.norm(module1.bias.data-module2.bias.data)**2
        elif 'BatchNorm' in str(type(module1)):
            norm_square_sum += torch.norm(module1.weight.data - module2.weight.data)**2
            norm_square_sum += torch.norm(module1.bias.data - module2.bias.data)**2
            #norm_square_sum += torch.norm(module1.running_mean - module2.running_mean)**2
            #norm_square_sum += torch.norm(module1.running_var - module2.running_var)**2
    #for (n1,p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
    #    if 'fc' in n1 or 'last' in n1:
    #        continue
    #    norm_square_sum += torch.norm(p1-p2)**2
    return torch.sqrt(norm_square_sum)
# Arguments
def main():
    args = get_args()

    # Seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True # Control randomness
    #device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    # torch.backends.cudnn.benchmark = False

    print('Load data...')
    data_dict = None
    dataset = data_handler.DatasetFactory.get_dataset('CIFAR100_for_Resnet')
    task_info = dataset.task_info
    print('\nTask info =', task_info)


    # Loader used for training data
    shuffle_idx = shuffle(np.arange(dataset.classes), random_state=args.seed)

    # list of dataloaders: it consists of dataloaders for each task
    train_dataset_loaders = data_handler.make_ContinualLoaders(dataset.train_data,
                                                            dataset.train_labels,
                                                            task_info,
                                                            transform=dataset.train_transform,
                                                            shuffle_idx = shuffle_idx,
                                                            data_dict = data_dict,
                                                           )

    test_dataset_loaders = data_handler.make_ContinualLoaders(dataset.test_data,
                                                           dataset.test_labels,
                                                           task_info,
                                                           transform=dataset.test_transform,
                                                           shuffle_idx = shuffle_idx,
                                                           data_dict = data_dict,
                                                          )

    # Get the required model
    myModel = networks.ModelFactory.get_model(args.model, task_info).to(device)
    test_model1 = networks.ModelFactory.get_model(args.model, task_info).to(device)
    test_model2 = networks.ModelFactory.get_model(args.model, task_info).to(device)
    #test_model1 = torchvision.models.resnet18(pretrained=False).to(device)
    #num_ftrs = test_model1.fc.in_features
    #test_model1.fc = nn.Linear(num_ftrs, 5)
    #test_model2 = torchvision.models.resnet18(pretrained=False).to(device)
    #test_model2.fc = nn.Linear(num_ftrs, 5)

    # Define the optimizer used in the experiment

    optimizer = torch.optim.Adam(myModel.parameters(), lr=args.lr, weight_decay=args.decay)

    # Initilize the evaluators used to measure the performance of the system.
    t_classifier = trainer.EvaluatorFactory.get_evaluator("trainedClassifier")

    ########################################################################################################################

    utils.print_model_report(myModel)
    utils.print_optimizer_config(optimizer)
    print('-' * 100)

    interpolation_range = np.arange(-1, 2, 0.1)
    # Loop tasks
    acc = np.zeros((len(task_info), len(task_info), len(interpolation_range)), dtype=np.float32)
    lss = np.zeros((len(task_info), len(task_info), len(interpolation_range)), dtype=np.float32)
    for t, ncla in task_info:
        if t == 0:
            continue
        print("tasknum:", t)
        # Add new classes to the train, and test iterator

        train_loader = train_dataset_loaders[t]
        test_loader = test_dataset_loaders[t]

        for u in range(t+1):
            test_loader = test_dataset_loaders[u]
            test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
            test_model1.load_state_dict(torch.load('./trained_model/_CIFAR100_for_Resnet_from_pretrained_resnet18_SGD_0_lamb_0_lr_0.1_batch_256_epoch_100_task_0.pt'))
            test_model2.load_state_dict(torch.load('./trained_model/_CIFAR100_for_Resnet_from_pretrained_resnet18_SGD_0_lamb_0_lr_0.1_batch_256_epoch_100_task_1.pt'))
            distance = compute_distance(test_model1, test_model2)
            print("Distance btw Two model : {}".format(distance))
            for i, lamb in enumerate(interpolation_range):
                for module, model1_module, model2_module in zip(myModel.modules(), test_model1.modules(), test_model2.modules()):
                    if 'Conv' in str(type(module)) or 'Linear' in str(type(module)):
                        module.weight.data = (lamb * model1_module.weight.data + (1-lamb) * model2_module.weight.data)
                        if module.bias is not None:
                            module.bias.data = (lamb * model1_module.bias.data + (1-lamb) * model2_module.bias.data)
                    elif 'BatchNorm' in str(type(module)):
                        module.weight.data = (lamb * model1_module.weight.data + (1-lamb) * model2_module.weight.data)
                        module.bias.data = (lamb * model1_module.bias.data + (1-lamb) * model2_module.bias.data)
                        module.running_mean.data = (lamb * model1_module.running_mean.data + (1-lamb) * model2_module.running_mean.data)
                        module.running_var.data = (lamb * model1_module.running_var.data + (1-lamb) * model2_module.running_var.data)
                test_loss, test_acc = t_classifier.evaluate(myModel, test_iterator, u, device)
                #test_loss, test_acc = t_classifier.one_task_evaluate(myModel, test_iterator, device)
                print('>>> Test on task {:2d}: Interpolation Coefficient : {:.2f}, loss={:.3f}, acc={:5.1f}% <<<'.format(u,lamb,  test_loss, 100 * test_acc))
                acc[t, u, i] = test_acc
                lss[t, u, i] = test_loss


        for task_t, acc_loss in enumerate(zip(acc, lss)):
            acc_t, loss_t = acc_loss
            np.savetxt('interpolate_acc_after_task_{}'.format(task_t), acc_t, '%.4f')
        if t==1:
            break


    #print('*' * 100)
    #print('Accuracies =')
    #for i in range(acc.shape[0]):
    #    print('\t', end='')
    #    for j in range(acc.shape[1]):
    #        print('{:5.1f}% '.format(100 * acc[i, j]), end='')
    #    print()
    #print('*' * 100)
    #print('Done!')


if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()

