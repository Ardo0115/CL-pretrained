from __future__ import print_function

import copy
import logging
from types import NoneType

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td
from PIL import Image
from tqdm import tqdm
import trainer

import networks
import torchvision

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, taskcla):
        super().__init__(model, args, optimizer, evaluator, taskcla)

        self.lamb=args.lamb


    def train(self, train_loader, test_loader, t, device = None):

        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t==0:
            model1 = self.train_from(None, train_loader, test_loader, t, from_pretrained = True)
            model2 = self.train_from(model1, train_loader, test_loader, t, from_pretrained = False)
            self.set_model_to_middle(model1, model2)
            self.set_radius(model1, model2)
            return
        else: # update fisher before start training new task
            self.update_frozen_model()
            self.update_fisher()


        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr)
        elif self.args.optimizer == 'SGD':
            self.optimzer = torch.optim.SGD(self.model.parameters(), lr=self.current_lr)
        
        # Now, you can update self.t
        self.t = t

        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)


        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.to(device), target.to(device)
                batch_size = data.shape[0]

                output = self.model(data)[t]
                loss_CE = self.criterion(output,target)


                self.optimizer.zero_grad()
                (loss_CE).backward()
                self.optimizer.step()

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t, self.device)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.evaluate(self.model, self.test_iterator, t, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()
        self.constrain_to_basin()

    def criterion(self,output,targets):
        """
        Arguments: output (The output logit of self.model), targets (Ground truth label)
        Return: loss function for the classification task

        """

        #######################################################################################



        # Write youre code here
        loss = self.ce
        return loss(output, targets)



        #######################################################################################

    def update_fisher(self):
        return None

    def train_from(self, from_model, train_loader, test_loader, t, from_pretrained = False):
        lr = self.args.lr
        self.setup_training(lr)
        device = self.device
        train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        if from_pretrained:
            model_to_train = torchvision.models.resnet18(pretrained=True)
            num_ftrs = model_to_train.fc.in_features
            model_to_train.fc = nn.Linear(num_ftrs, self.task_info[0][1])
        else:
            model_to_train = copy.deepcopy(from_model)
        model_to_train.to(device)

        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(model_to_train.parameters(), lr=self.current_lr)
        elif self.args.optimizer == 'SGD':
            self.optimzer = torch.optim.SGD(model_to_train.parameters(), lr=self.current_lr)

        for epoch in range(self.args.nepochs):
            model_to_train.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(train_iterator):
                data, target = samples
                data, target = data.to(device), target.to(device)
                batch_size = data.shape[0]

                output = model_to_train(data)
                loss_CE = self.criterion(output,target)


                self.optimizer.zero_grad()
                (loss_CE).backward()
                self.optimizer.step()

            train_loss,train_acc = self.evaluator.evaluate(model_to_train, train_iterator, t, self.device)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.evaluate(model_to_train, test_iterator, t, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()

    def set_model_to_middle(self, model1, model2):
        for my_module, module1, module2 in zip(self.model.modules(), model1.modules(), model2.modules()):
            if 'Conv' in str(type(my_module)):
                my_module.weight.data = (module1.weight.data + module2.weight.data) / 2.0
                if my_module.bias is not None: 
                    my_module.bias.data = (module1.bias.data + module2.bias.data) / 2.0
            elif 'BatchNorm' in str(type(my_module)):
                my_module.weight.data = (module1.weight.data + module2.weight.data) / 2.0
                my_module.bias.data = (module1.bias.data + module2.bias.data) / 2.0
                my_module.running_mean = (module1.running_mean + module2.running_mean) / 2.0
                my_module.running_var = (module1.running_var + module2.running_var) / 2.0
        self.center = copy.deepcopy(self.model)
        """
        MUST IMPLEMENT LAST LINEAR LAYER
        """
    
    def set_radius(self, model1, model2):
        param1_flatten = torch.Tensor().to(self.device)
        param2_flatten = torch.Tensor().to(self.device)
        for param1, param2 in zip(model1.paramters(), model2.paramters()):
            param1_flatten = torch.cat([param1_flatten, param1.view(-1)])
            param2_flatten = torch.cat([param2_flatten, param2.view(-1)])
        
        self.radius = torch.norm(param1_flatten-param2_flatten) / 2.0

    def constrain_to_basin(self):
        # param1_flatten = torch.Tensor().to(self.device)
        # param2_flatten = torch.Tensor().to(self.device)
        # for param1, param2 in zip(model1.paramters(), model2.paramters()):
        #     param1_flatten = torch.cat([param1_flatten, param1.view(-1)])
        #     param2_flatten = torch.cat([param2_flatten, param2.view(-1)])
        
