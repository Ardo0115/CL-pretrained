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
        self.t = -1

    def train(self, train_loader, train_loader_CIFAR10, test_loader, t, device = None):

        log_name = ''
        log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format('', 'CIFAR10', 'ewc_resnet18_Adam',0,
                                                                        0.0, 0.001, 256, 60)
        #log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(self.args.date, self.args.dataset, self.args.trainer, self.args.seed,
        #                                                                0.0, 0.001, self.args.batch_size, self.args.nepochs)
        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t == 0:
            tmp_model = networks.ModelFactory.get_model('CIFAR10', 'ewc_resnet18_Adam', [(0,10)])

            trained_model_fname = './trained_model/' + log_name + '_task_0.pt'
            #model_pretrained = torch.load(trained_model_fname)
            tmp_model.load_state_dict(torch.load(trained_model_fname))
            tmp_model.to(self.device)

            #for n,p in self.model.named_parameters():
            #    if n in model_pretrained.keys() and 'last' not in n:
            #        p.data.copy_(model_pretrained[n].data)
            for module, module_pretrained in zip(self.model.modules(), tmp_model.modules()):
                if 'Conv' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    if module.bias is not None: 
                        module.bias.data.copy_(module_pretrained.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    module.running_mean.copy_(module_pretrained.running_mean)
                    module.running_var.copy_(module_pretrained.running_var)
            self.fisher_iterator = torch.utils.data.DataLoader(train_loader_CIFAR10, batch_size=20, shuffle=True)
            self.update_frozen_model()
            #self.unfreeze_film()
            #self.freeze_except_film_and_last()
            self.update_fisher()
            '''
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
            #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr)
            '''
        else:
            self.update_frozen_model()
            self.update_fisher()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr)
        '''
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
        self.update_frozen_model()
        if t > 1:
            self.update_fisher()
        '''

        # Now, you can update self.t
        self.t = t

        # There are data for task 't' in train_loader
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)

        for epoch in range(self.args.nepochs):
            #self.model.train()
            self.train_nobn(self.model)
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

    def criterion(self,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if self.t>-1:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
                if 'last' not in name:
                    loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        return self.ce(output,targets)+self.lamb*loss_reg

    def compute_diag_fisher(self):
        if self.t == -1:
            # Init
            log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format('', 'CIFAR10', 'ewc_resnet18_Adam',0,
                                                                            0.0, 0.001, 256, 60)
            tmp_model = networks.ModelFactory.get_model('CIFAR10', 'ewc_resnet18_Adam', [(0,10)])

            trained_model_fname = './trained_model/' + log_name + '_task_0.pt'
            #model_pretrained = torch.load(trained_model_fname)
            tmp_model.load_state_dict(torch.load(trained_model_fname))
            tmp_model.to(self.device)
            fisher={}
            for n,p in self.model.named_parameters():
                if 'last' not in n:
                    fisher[n]=0*p.data
            # Compute
            #self.model.train()
            self.train_nobn(self.model)
            criterion = torch.nn.CrossEntropyLoss()
            for samples in tqdm(self.fisher_iterator):
                data, target = samples
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]

                # Forward and backward
                #self.model.zero_grad()
                tmp_model.zero_grad()
                outputs = tmp_model.forward(data)[self.t]
                loss=self.criterion(outputs, target)
                loss.backward()

                # Get gradients
                for n,p in tmp_model.named_parameters():
                    if 'last' not in n:
                        if p.grad is not None:
                            fisher[n]+=batch_size**2*p.grad.data.pow(2) # self.args.batch_Size? this is different from 20?
            # Mean
            with torch.no_grad():
                for n,_ in tmp_model.named_parameters():
                    if 'last' not in n:
                        fisher[n]=fisher[n]/len(self.fisher_iterator.dataset)
        else:
            # Init
            fisher={}
            for n,p in self.model.named_parameters():
                if 'last' not in n:
                    fisher[n]=0*p.data
            # Compute
            #self.model.train()
            self.train_nobn(self.model)
            criterion = torch.nn.CrossEntropyLoss()
            for samples in tqdm(self.fisher_iterator):
                data, target = samples
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]

                # Forward and backward
                #self.model.zero_grad()
                self.model.zero_grad()
                outputs = self.model.forward(data)[self.t]
                loss=self.criterion(outputs, target)
                loss.backward()

                # Get gradients
                for n,p in self.model.named_parameters():
                    if 'last' not in n:
                        if p.grad is not None:
                            fisher[n]+=batch_size**2*p.grad.data.pow(2) # self.args.batch_Size? this is different from 20?
            # Mean
            with torch.no_grad():
                for n,_ in self.model.named_parameters():
                    if 'last' not in n:
                        fisher[n]=fisher[n]/len(self.fisher_iterator.dataset)
        return fisher


    def update_fisher(self):
        if self.t>-1:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                if 'last' not in n:
                    fisher_old[n]=self.fisher[n].clone()
        self.fisher=self.compute_diag_fisher()
        if self.t>-1:
            for n,_ in self.model.named_parameters():
                if 'last' not in n:
                    self.fisher[n]=(self.fisher[n]+fisher_old[n]*(self.t+1))/(self.t+2)
