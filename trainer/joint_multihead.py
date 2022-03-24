from __future__ import print_function

import copy
import logging
from ntpath import join

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
        if t == 0:
            joint_singlehead_model = networks.ModelFactory.get_model(self.args.model, [(0,100)]).to(device)
            joint_singlehead_model.load_state_dict(torch.load('./trained_model/_CIFAR100_for_Resnet_joint_Adam_0_lamb_lr_0.001_batch_256_epoch_60_task_0.pt'))
            for module, module_pretrained in zip(self.model.modules(), joint_singlehead_model.modules()):
                if 'Conv' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    if module.bias is not None: 
                        module.bias.data.copy_(module_pretrained.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    module.running_mean.copy_(module_pretrained.running_mean)
                    module.running_var.copy_(module_pretrained.running_var)
            # self.update_frozen_model()
            

        # else:
            # self.update_frozen_model()
            # self.update_fisher()
            
        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr)
        elif self.args.optimizer == 'SGD':
            self.optimzer = torch.optim.SGD(self.model.parameters(), lr=self.current_lr)
        # Now, you can update self.t
        
        # Only train classifier
        for n, p in self.model.named_parameters():
                if 'last' not in n:
                    p.requires_grad = False
        self.t = t

        # There are data for task 't' in train_loader
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        # self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)

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
        log_name = '_{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(self.args.dataset,self.args.trainer, self.args.optimizer, self.args.seed,
                                                                           self.args.lamb, self.args.lr, self.args.batch_size, self.args.nepochs)
        torch.save(self.model.state_dict(), './trained_model/' + log_name + '_task_{}.pt'.format(t))

    def criterion(self,output,targets):
        # Regularization for all previous tasks
        # loss_reg=0
        # if self.t>0:
        #     for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
        #         loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        loss_ce = self.ce(output, targets)
        # print("loss_ce : {}, loss_reg : {}, ce/reg : {}".format(loss_ce, loss_reg, loss_ce/loss_reg))

        return loss_ce

    def compute_diag_fisher(self):
        # Init
        fisher={}
        for n,p in self.model.named_parameters():
            fisher[n]=0*p.data
        # Compute
        self.model.eval()
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

            # Get gradients
            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n]+=batch_size**2*p.grad.data.pow(2) # self.args.batch_Size? this is different from 20?
        # f = open("{}/knowledge_fisher/task_{}_knowledge_{}.txt".format(os.getcwd(),self.t, self.knowledge_ratio), 'a')
        # Mean
        with torch.no_grad():
            for n,_ in self.model.named_parameters():
                fisher[n]=fisher[n]/len(self.fisher_iterator.dataset)
                # f.write("fisher[{}] : {}\n".format(n, torch.norm(fisher[n])))
        # f.close()
        return fisher


    def update_fisher(self):
        if self.t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=self.compute_diag_fisher()
        if self.t>0:
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*(self.t))/(self.t+1)
