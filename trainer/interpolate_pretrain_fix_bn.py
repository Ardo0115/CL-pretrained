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
import torchvision

class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, task_info):
        super().__init__(model, args, optimizer, evaluator, task_info)

        self.lamb=args.lamb
        self.knowledge_ratio = args.knowledge_ratio

    def train(self, train_loader, test_loader, t, device = None):
        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t > 0:
            self.update_frozen_model()
        
        tmp_model = torchvision.models.resnet18(pretrained=True)
        for module, module_pretrained in zip(self.model.modules(), tmp_model.modules()):
            if 'Conv' in str(type(module)):
                module.weight.data.copy_(module_pretrained.weight.data)
                if module.bias is not None:
                    module.bias.data.copy_(module_pretrained.weight.data)
            # elif 'BatchNorm' in str(type(module)):
            #     module.weight.data.copy_(module_pretrained.weight.data)
            #     module.bias.data.copy_(module_pretrained.bias.data)
            #     module.running_mean.copy_(module_pretrained.running_mean)
            #     module.running_var.copy_(module_pretrained.running_var)

        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr)
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_lr)
        # Now, you can update self.t
        self.t = t

        # There are data for task 't' in train_loader
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)

        for epoch in range(self.args.nepochs):
            self.model.eval()
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
        log_name = '{}_{}_{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(self.args.date, self.args.dataset, 'from_pretrained', self.args.model, self.args.optimizer, self.args.seed,self.args.lamb, self.args.lr, self.args.batch_size, self.args.nepochs)
        torch.save(self.model.state_dict(), './trained_model/' + log_name + '_task_{}.pt'.format(t))
        if self.t > 0:
            for module, module_old in zip(self.model.modules(), self.model_fixed.modules()):
                if 'Conv' in str(type(module)):
                    curr_weight = module.weight.data
                    module.weight.data = (curr_weight + module_old.weight.data*self.t) / (self.t+1)
                    if module.bias is not None:
                        curr_bias = module.bias.data
                        module.bias.data = (curr_bias + module_old.bias.data*self.t) / (self.t+1)
                # elif 'BatchNorm' in str(type(module)):
                #     curr_weight = module.weight.data
                #     curr_bias = module.bias.data
                #     curr_running_mean = module.running_mean.data
                #     curr_running_var = module.running_var.data
                #     module.weight.data = (curr_weight + module_old.weight.data*self.t) / (self.t+1)
                #     module.bias.data = (curr_bias + module_old.bias.data*self.t) / (self.t+1)
                #     module.running_mean.data = (curr_running_mean + module_old.running_mean.data*self.t) / (self.t+1)
                #     module.running_var.data = (curr_running_var + module_old.running_var.data*self.t) / (self.t+1)
    def criterion(self,output,targets):
        # Regularization for all previous tasks
        loss_ce = self.ce(output, targets)
        return loss_ce
