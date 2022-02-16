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

# Arguments
def main():
    args = get_args()

    #########################################################################################################################

    log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.dataset, args.trainer,args.seed,
                                                                           args.lamb, args.lr, args.batch_size, args.nepochs)

    if args.output == '':
        args.output = './result_data_new/' + log_name + '.txt'
    ########################################################################################################################
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

    if not os.path.isdir('dat'):
        print('Make directory for dataset')
        os.makedirs('dat')

    print('Load data...')
    data_dict = None
    dataset = data_handler.DatasetFactory.get_dataset(args.dataset)
    task_info = dataset.task_info
    print('\nTask info =', task_info)

    if not os.path.isdir('result_data_new'):
        print('Make directory for saving results')
        os.makedirs('result_data_new')

    if not os.path.isdir('trained_model'):
        print('Make directory for saving trained models')
        os.makedirs('trained_model')

    # Args -- Experiment

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
    myModel = networks.ModelFactory.get_model(args.dataset, args.trainer, task_info).to(device)

    # Define the optimizer used in the experiment

    optimizer = torch.optim.Adam(myModel.parameters(), lr=args.lr, weight_decay=args.decay)

    # Initilize the evaluators used to measure the performance of the system.
    t_classifier = trainer.EvaluatorFactory.get_evaluator("trainedClassifier")

    # Trainer object used for training
    myTrainer = trainer.TrainerFactory.get_trainer(myModel, args, optimizer, t_classifier, task_info)

    ########################################################################################################################

    utils.print_model_report(myModel)
    utils.print_optimizer_config(optimizer)
    print('-' * 100)

    # Loop tasks
    acc = np.zeros((len(task_info), len(task_info)), dtype=np.float32)
    lss = np.zeros((len(task_info), len(task_info)), dtype=np.float32)
    for t, ncla in task_info:
        print("tasknum:", t)
        # Add new classes to the train, and test iterator

        train_loader = train_dataset_loaders[t]
        test_loader = test_dataset_loaders[t]

        myTrainer.train(train_loader, test_loader, t, device)


        for u in range(t+1):
            test_loader = test_dataset_loaders[u]
            test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
            if u == t:
                test_model = myTrainer.model
            else:
                test_model = networks.ModelFactory.get_model(args.dataset, args.trainer, task_info).to(device)
                early_model = torch.load('./trained_model/' + log_name + '_task_{}.pt'.format(u))
                #test_model.load_state_dict(torch.load('./trained_model/' + log_name + '_task_{}.pt'.format(u)))
                test_model.load_state_dict(copy.deepcopy(myTrainer.model.state_dict()))
                #test_model.load_state_dict(early_model)
                for n, p in test_model.named_parameters():
                    if 'mask' in n:
                        p.data.copy_(early_model[n].data)
                #for n, n_curr in zip(test_model.state_dict().keys(), early_model.keys()):
                #    if 'last' not in n:
                #        assert (test_model.state_dict()[n].data - early_model[n_curr].data).abs().sum()<1e-8
                #for n, n_curr in zip(test_model.state_dict().keys(), myTrainer.model.state_dict().keys()):
                #    if 'mask' not in n:
                #        assert (test_model.state_dict()[n].data - myTrainer.model.state_dict()[n_curr].data).abs().sum() <1e-8
                #        print("n: {}, n_curr : {} are different!".format(n, n_curr))
                    #if 'running_mean' in n:
                    #    print("n: {}, n_curr : {} are running_mean!".format(test_model.state_dict()[n].data, myTrainer.model.state_dict()[n_curr].data))
                    #p = 0
                    #if 'last' in n or 'mask' in n:
                    #    continue
                    #elif 'running_mean' in n or 'running_var' in n:
                    #    p.data.copy_(myTrainer.model.state_dict()[n].data)
                    #else:
                    #    assert (p.data - myTrainer.model.state_dict()[n].data).abs().sum() < 1e-8
            test_loss, test_acc = t_classifier.evaluate(test_model, test_iterator, u, device)
            print('>>> Test on task {:2d}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loss, 100 * test_acc))
            acc[t, u] = test_acc
            lss[t, u] = test_loss

        print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[t,:t+1])))

        print('Save at ' + args.output)
        np.savetxt(args.output, acc, '%.4f')
        if args.lamb == 0.0 and args.lr == 0.001:
            torch.save(myModel.state_dict(), './trained_model/' + log_name + '_task_{}.pt'.format(t))


    print('*' * 100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end='')
        for j in range(acc.shape[1]):
            print('{:5.1f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*' * 100)
    print('Done!')


if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
