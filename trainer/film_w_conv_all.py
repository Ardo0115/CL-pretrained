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
        self.lamb2=args.lamb2


    def train(self, train_loader, test_loader, t, device = None):

        log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(self.args.date, self.args.dataset, 'film_w_conv',self.args.seed,
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
        elif t == 1: # update fisher before starting training new task
            self.update_frozen_model()
            self.unfreeze_film()
            #self.freeze_except_film_and_last()
            #self.update_fisher()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr, weight_decay=self.args.decay)
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

    def freeze_except_film_and_last(self):
        matches = ['film', 'last', 'conv4', 'batchnorm4']
        for (name, param) in self.model.named_parameters():
            if all(x not in name for x in matches):
                    param.requires_grad = False

    def get_film_params(self, model):
        film_params = torch.Tensor().to(self.device)
        for (name, param) in model.named_parameters():
            if "film" in name:
                film_params = torch.cat([film_params, param.view(-1)])
        return film_params

    def get_conv_params(self, model):
        conv_params = torch.Tensor().to(self.device)
        for (name, param) in model.named_parameters():
            if "conv4" in name or "batchnorm4" in name:
                conv_params = torch.cat([conv_params, param.view(-1)])
        return conv_params

    '''Given Solution'''
    def criterion(self,output,targets):
        # Concat all gammas, betas in film layers
        #for (name, param) in self.model.named_parameters():
        #    if "film" in name:
        new_film_params = self.get_film_params(self.model)
        old_film_params = self.get_film_params(self.model_fixed)
        loss_reg_film = 0
        loss_reg_conv = 0
        if self.t > 1:
            film_param_diff = new_film_params - old_film_params
            film_fisher_param_diff_mul = torch.matmul(self.film_fisher, film_param_diff)

            loss_reg_film += torch.matmul(film_param_diff, film_fisher_param_diff_mul) / 2.0
            for (name, param), (_,param_old) in zip(self.model.named_parameters(), self.model_fixed.named_parameters()):
                loss_reg_conv+=torch.sum(self.conv_fisher[name]*(param_old-param).pow(2)) / 2.0

        # Regularization for all previous tasks
        # loss_reg = 0
        # if self.t > 0:
        #     for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
        #         loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        return self.ce(output,targets)+self.lamb*loss_reg_film + self.lamb2*loss_reg_conv

    def compute_film_fisher(self):
        # Init
        conv_fisher={}
        for n,p in self.model.named_parameters():
            conv_fisher[n]=0*p.data
        film_fisher = 0
        # Compute
        self.model.train()
        ##### Must Check CNN layers are frozen ###
        criterion = torch.nn.CrossEntropyLoss()
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
                if "film" not in n and p.grad is not None:
                    conv_fisher[n]+=batch_size**2*p.grad.data.pow(2)
            film_fisher += batch_size**2 * torch.outer(film_params_grad, film_params_grad)

        # Mean
        with torch.no_grad():
            film_fisher /= len(self.fisher_iterator.dataset)
            for n,_ in self.model.named_parameters():
                conv_fisher[n] = conv_fisher[n]/len(self.fisher_iterator.dataset)
        return film_fisher, conv_fisher

    def update_fisher(self):
        # if self.t>0:
        #     fisher_old={}
        #     for n,_ in self.model.named_parameters():
        #         fisher_old[n]=self.fisher[n].clone()
        if self.t>1:
            film_fisher_old = (self.film_fisher).clone()
            conv_fisher_old = {}
            for n,_ in self.model.named_parameters():
                if 'film' not in n:
                    conv_fisher_old[n]=self.conv_fisher[n].clone()
        self.film_fisher, self.conv_fisher = self.compute_film_fisher()
        if self.t>1:
            self.film_fisher=(self.film_fisher+film_fisher_old*(self.t-1))/(self.t)
            for n, _ in self.model.named_parameters():
                if 'film' not in n:
                    self.conv_fisher[n]=(self.conv_fisher[n]+conv_fisher_old[n]*(self.t-1))/(self.t)
