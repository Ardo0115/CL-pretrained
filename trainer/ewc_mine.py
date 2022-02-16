from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td
from PIL import Image
from tqdm import tqdm
import trainer

import networks

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, task_info):
        super().__init__(model, args, optimizer, evaluator, task_info)

        self.lamb=args.lamb


    def train(self, train_loader, test_loader, t, device = None):

        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t>0: # update fisher before starting training new task
            self.update_frozen_model()
            self.update_fisher()

        # Now, you can update self.t
        self.t = t

        # There are data for task 't' in train_loader
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)


        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            # samples : [data, target]
            # data.shape : (batch_size, 32, 32, 3) for CIFAR100
            # target.shape : (batch_size)
            # len(train_iterator) : 10000 / 256(batch_size) = 39+a => 40 for CIFAR100
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.to(device), target.to(device)
                batch_size = data.shape[0]

                # Reason why model(data)[t] : The last layer is ModuleList
                # t-th layer in ModuleList is for t-th task
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

    def criterion(self,output,targets):
        """
        Arguments: output (The output logit of self.model), targets (Ground truth label)
        Return: loss function for the regularization-based continual learning

        For the hyperparameter on regularization, please use self.lamb
        """

        #######################################################################################



        # Write youre code here
        loss = self.ce(output,targets)
        if self.t != 0:
            for j, (param, param_fixed) in enumerate(zip(self.model.parameters(), self.model_fixed.parameters())):
                fisher_diag_sum = 0
                for task_t, fisher in self.fisher.items():
                    fisher_diag_sum += fisher[j]
                loss += self.lamb/2.0*torch.sum((1+fisher_diag_sum)*(param.view(-1)-param_fixed.view(-1))**2)
        return loss



        #######################################################################################

    def compute_diag_fisher(self):
        """
        Arguments: None. Just use global variables (self.model, self.criterion, ...)
        Return: Diagonal Fisher matrix.

        This function will be used in the function 'update_fisher'
        """


        #######################################################################################



        # Write youre code here
        grads = []
        for _, param in enumerate(self.model.parameters()):
            grads.append(torch.zeros_like(param.view(-1)))
        for samples in self.fisher_iterator:
            data, target = samples
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]

            output = self.model(data)[self.t]
            loss_for_fisher = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss_for_fisher.backward()
            for i, param in enumerate(self.model.parameters()):
                if param.grad is None:
                    continue
                grads[i] = grads[i] + param.grad.view(-1)

        fisher_diag = list(map(lambda x:x**2, grads))
        return fisher_diag

        #######################################################################################

    def update_fisher(self):

        """
        Arguments: None. Just use global variables (self.model, self.fisher, ...)
        Return: None. Just update the global variable self.fisher
        Use 'compute_diag_fisher' to compute the fisher matrix
        """

        #######################################################################################



        # Write youre code here
        self.fisher[self.t] = self.compute_diag_fisher()


        #######################################################################################
