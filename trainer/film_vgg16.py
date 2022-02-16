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
import sys

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, task_info):
        super().__init__(model, args, optimizer, evaluator, task_info)

        self.lamb=args.lamb


    def train(self, train_loader, test_loader, t, device = None):

        log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format('', 'CIFAR10', 'film_vgg16',0,
                                                                        0.0, 0.001, 256, 60)
        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t == 0: 
            
            # Load cifar10 pretrained model
            trained_model_fname = './trained_model/' + log_name + '_task_0.pt'
            #self.model.load_state_dict(torch.load(trained_model_fname))
            model_pretrained = torch.load(trained_model_fname)
            for n,p in self.model.named_parameters():
                if n in model_pretrained.keys() and 'last' not in n:
                    p.data.copy_(model_pretrained[n].data)

            # And, Only train film network
            self.update_frozen_model()
            self.unfreeze_film()
            self.freeze_except_film_and_last()
            #self.update_fisher()
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr)
        else:
            self.update_frozen_model()
            self.update_fisher()

        # Now, you can update self.t
        self.t = t

        # There are data for task 't' in train_loader
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)

        for epoch in range(self.args.nepochs):
            self.train_nobn(self.model)
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
            if np.isnan(train_loss):
                sys.exit()
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.evaluate(self.model, self.test_iterator, t, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()

    def train_nobn(self, model):
        model.train()

        for module in model.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def freeze_film(self):
        for (name, param) in self.model.named_parameters():
            if "gamma" in name or 'beta' in name:
                param.requires_grad = False
    def unfreeze_film(self):
        for (name, param) in self.model.named_parameters():
            if "gamma" in name or 'beta' in name:
                param.requires_grad = True

    def freeze_except_film_and_last(self):
        matches = ['gamma', 'beta', 'last']
        for (name, param) in self.model.named_parameters():
            if all(x not in name for x in matches):
                    param.requires_grad = False

    def get_film_params(self, model):
        film_params = torch.Tensor().to(self.device)
        for (name, param) in model.named_parameters():
            if "gamma" in name or 'beta' in name:
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
            loss_reg += torch.matmul(param_diff, fisher_param_diff_mul) / 2.0

        # Regularization for all previous tasks
        # loss_reg = 0
        # if self.t > 0:
        #     for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
        #         loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        return self.ce(output,targets)+self.lamb*loss_reg

    def compute_film_fisher(self):
        # Init
        # for n,p in self.model.named_parameters():
        #     fisher[n]=0*p.data
        # Compute
        self.train_nobn(self.model)
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
                if ("gamma" in n or 'beta' in n)  and p.grad is not None:
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
            self.fisher=(self.fisher+fisher_old*(self.t))/(self.t+1)
