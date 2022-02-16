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

import os.path

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, task_info):
        super().__init__(model, args, optimizer, evaluator, task_info)

        self.lamb=args.lamb


    def train(self, train_loader, test_loader, t, device = None):

        log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(self.args.date, self.args.dataset, self.args.trainer,self.args.seed,
                                                                        0.0, 0.001, self.args.batch_size, self.args.nepochs)
        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t == 0: # Only train backbone network
            trained_model_fname = './trained_model/' + log_name + '_task_0.pt'
            if os.path.isfile(trained_model_fname):
                self.model.load_state_dict(torch.load(trained_model_fname))
                self.t = t
                # There are data for task 't' in train_loader
                self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
                self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
                self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)
                return
            else:
                self.freeze_film()
        # Now, only train film network
        else: # update fisher before starting training new task
            self.update_frozen_model()
            self.unfreeze_film()
            self.freeze_except_film_and_last(t)
            self.update_fisher()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr, weight_decay=self.args.decay)

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
                #if self.t > 0:
                #    conv_param_dict_before = {}
                #    for (name, param) in self.model.named_parameters():
                #        if "film" not in name:
                #            assert param.requires_grad == False
                #            conv_param_dict_before[name] = param.clone().detach()
                #        else:
                #            assert param.requires_grad == True
                self.optimizer.step()
                #if self.t > 0:
                #    conv_param_dict_after = {}
                #    for (name, param) in self.model.named_parameters():
                #        if "film" not in name:
                #            assert param.requires_grad == False
                #            conv_param_dict_after[name] = param.clone().detach()
                #        else:
                #            assert param.requires_grad == True
                #    for key in conv_param_dict_after:
                #        assert torch.eq(conv_param_dict_after[key].data, conv_param_dict_before[key].data)

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t, self.device)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.evaluate(self.model, self.test_iterator, t, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()

    def freeze_film(self):
        for (name, param) in self.model.named_parameters():
            if "film" in name:
                param.requires_grad = False
    def unfreeze_film(self):
        for (name, param) in self.model.named_parameters():
            if "film" in name:
                param.requires_grad = True

    def freeze_except_film_and_last(self, t):
        matches = ['film']
        for i in range(t, self.args.tasknum):
            matches.append('last.{}'.format(i))

        for (name, param) in self.model.named_parameters():
            if all(x not in name for x in matches):
                    param.requires_grad = False

    def get_film_params(self, model):
        film_params = torch.Tensor().to(self.device)
        for (name, param) in model.named_parameters():
            if "film" in name:
                film_params = torch.cat([film_params, param.view(-1)])
        return film_params

    '''Given Solution'''
    def criterion(self,output,targets):
        # Concat all gammas, betas in film layers
        #for (name, param) in self.model.named_parameters():
        #    if "film" in name:
        new_film_params = self.get_film_params(self.model)
        old_film_params = self.get_film_params(self.model_fixed)
        loss_reg = 0
        if self.t > 0:
            param_diff = new_film_params - old_film_params
            fisher_param_diff_mul = torch.matmul(self.fisher, param_diff)
            loss_reg += torch.matmul(param_diff, fisher_param_diff_mul)

        # Regularization for all previous tasks
        # loss_reg = 0
        # if self.t > 0:
        #     for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
        #         loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        return self.ce(output,targets)+self.lamb*loss_reg / 2.0

    def compute_film_fisher(self):
        # Init
        # for n,p in self.model.named_parameters():
        #     fisher[n]=0*p.data
        # Compute
        self.model.train()
        ##### Must Check CNN layers are frozen ###
        criterion = torch.nn.CrossEntropyLoss()
        fisher = 0
        for samples in tqdm(self.fisher_iterator):
            data, target = samples
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]
            # Forward and backward
            self.model.zero_grad()
            outputs = self.model.forward(data)[self.t]
            loss=self.criterion(outputs, target)
            loss.backward()

            film_params_grad = torch.Tensor().to(self.device) # device?
            # Get gradients
            for n,p in self.model.named_parameters():
                if "film" in n and p.grad is not None:
                    film_params_grad = torch.cat([film_params_grad, p.grad.view(-1)])
            fisher += batch_size**2 * torch.outer(film_params_grad, film_params_grad)
            

        with torch.no_grad():
            fisher /= len(self.fisher_iterator.dataset)
        # with torch.no_grad(): # needs?
        #     for n,_ in self.model.named_parameters():
        #         fisher[n]=fisher[n]/len(self.train_iterator) # self.fisher_iterator?
        return fisher

    def update_fisher(self):
        # if self.t>0:
        #     fisher_old={}
        #     for n,_ in self.model.named_parameters():
        #         fisher_old[n]=self.fisher[n].clone()
        if self.t>0:
            fisher_old = (self.fisher).clone()
        self.fisher = self.compute_film_fisher()
        if self.t>0:
            self.fisher=(self.fisher+fisher_old*self.t)/(self.t+1)
    '''
    My Solution
    '''
    '''
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


'''
