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

import matplotlib.pyplot as plt


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, taskcla):
        super().__init__(model, args, optimizer, evaluator, taskcla)

        self.lamb=args.lamb
        self.tilt=args.grad_tilt

    def get_tilted(self, vector):
        x = torch.rand(vector.size())
        if torch.equal(vector, torch.zeros_like(vector)):
            return vector
        assert x.size() == vector.size()
        x = x.view(-1)
       # unit_v = vector.view(-1) / torch.linalg.norm(vector.view(-1))
        x -= torch.dot(x, vector.view(-1)) / torch.dot(vector.view(-1), vector.view(-1)) * vector.view(-1)
       # print(abs(torch.dot(x.view(-1), vector.view(-1))))
        #assert abs(torch.dot(x, vector.view(-1))) < 1e-3
        x /= torch.norm(x)
        x *= torch.norm(vector.view(-1))
        tilt_rad = np.pi * self.tilt / 10.0
        tilted_vec = (np.sin(tilt_rad)*x+np.cos(tilt_rad)*vector.view(-1))
        tilted_vec = tilted_vec.view(vector.size())
        return tilted_vec

    def train(self, train_loader, test_loader, t, device = None):

        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t>0: # update fisher before start training new task
            self.update_frozen_model()
            self.update_fisher()

        # Now, you can update self.t
        self.t = t

        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)


        loss_history = []
        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.to(device), target.to(device)
                batch_size = data.shape[0]

                output = self.model(data)[t]
                loss_CE = self.criterion(output,target)
                loss_history.append(loss_CE.data)


                self.optimizer.zero_grad()
                (loss_CE).backward()

                #for parameter in self.model.parameters():
                #    if parameter.grad is not None:
                #       # print("original grad : {}".format(parameter.grad.data))
                #        parameter.grad = self.get_tilted(parameter.grad)
                #       #  print("Changed grad : {}".format(parameter.grad.data))
                if epoch >= self.args.nepochs / 2:
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                           # print("original grad : {}".format(parameter.grad.data))
                            parameter.grad = self.get_tilted(parameter.grad)
                           #  print("Changed grad : {}".format(parameter.grad.data))
                self.optimizer.step()

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t, self.device)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.evaluate(self.model, self.test_iterator, t, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()
        return loss_history

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
