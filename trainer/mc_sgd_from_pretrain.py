from __future__ import print_function

import copy
from curses import start_color
import logging
from random import sample

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
        self.memory = {}

    def train(self, train_dataset_loaders, test_dataset_loaders, t, device = None):
        train_loader = train_dataset_loaders[t]
        test_loader = test_dataset_loaders[t]
        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t > 0:
            self.update_frozen_model()
        if t ==0:
            pretrained_model = torchvision.models.resnet18(pretrained=True)
            for module, module_pretrained in zip(self.model.modules(), pretrained_model.modules()):
                if 'Conv' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    if module.bias is not None:
                        module.bias.data.copy_(module_pretrained.weight.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    module.running_mean.copy_(module_pretrained.running_mean)
                    module.running_var.copy_(module_pretrained.running_var)

        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr)
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_lr)
        # Now, you can update self.t
        self.t = t

        # There are data for task 't' in train_loader
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.appendMemory(train_loader)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)

        for epoch in range(self.args.nepochs):
            self.model.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(self.train_iterator):
                data, target = samples
                data, target = data.to(device), target.to(device)
                batch_size = data.shape[0]

                output = self.model(data, t)
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
        log_name = '{}_{}_{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}_tuningepoch_{}_linesamples_{}_tasknum_{}'.format(self.args.date, self.args.dataset, 'from_previous', self.args.model, self.args.optimizer, self.args.seed,self.args.lamb, self.args.lr, self.args.batch_size, self.args.nepochs, self.args.tuning_epochs, self.args.line_samples, self.args.tasknum)
        torch.save(self.model.state_dict(), './trained_model/' + log_name + '_task_{}.pt'.format(t))
        if self.t > 0:
            tmp_model = copy.deepcopy(self.model)
            # tmp_model = torchvision.models.resnet18(pretrained=False)
            # for module_pretrained, module in zip(self.model.modules(), tmp_model.modules()):
            #     if 'Conv' in str(type(module)) or 'Linear' in str(type(module)):
            #         module.weight.data.copy_(module_pretrained.weight.data)
            #         if module.bias is not None:
            #             module.bias.data.copy_(module_pretrained.weight.data)
            #     elif 'BatchNorm' in str(type(module)):
            #         module.weight.data.copy_(module_pretrained.weight.data)
            #         module.bias.data.copy_(module_pretrained.bias.data)
            #         module.running_mean.copy_(module_pretrained.running_mean)
            #         module.running_var.copy_(module_pretrained.running_var)
            lamb=0.5
            for module, model1_module, model2_module in zip(self.model.modules(), self.model_fixed.modules(), tmp_model.modules()):
                if 'Conv' in str(type(module)) or 'Linear' in str(type(module)):
                    module.weight.data = ((1-lamb) * model1_module.weight.data + lamb * model2_module.weight.data)
                    if module.bias is not None:
                        module.bias.data = ((1-lamb) * model1_module.bias.data + lamb * model2_module.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data = ((1-lamb) * model1_module.weight.data + lamb * model2_module.weight.data)
                    module.bias.data = ((1-lamb) * model1_module.bias.data + lamb * model2_module.bias.data)
                    module.running_mean.data = ((1-lamb) * model1_module.running_mean.data + lamb * model2_module.running_mean.data)
                    module.running_var.data = ((1-lamb) * model1_module.running_var.data + lamb * model2_module.running_var.data)
                    module.num_batches_tracked = ((1-lamb) * model1_module.num_batches_tracked.data + lamb * model2_module.num_batches_tracked.data)


            # lr = self.args.lr
            lr = 0.01
            self.setup_training(lr)
            if self.args.optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr)
            elif self.args.optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_lr)
            for epoch in range(self.args.tuning_epochs):
                self.model.train()
                self.update_lr(epoch, self.args.schedule)
            
                
                n_line_samples = self.args.line_samples
                alphas = [float(i)/n_line_samples for i in range(1, n_line_samples)]
                self.optimizer.zero_grad()
                for alpha in alphas:
                    sampled_model = networks.ModelFactory.get_model(self.args.model, self.task_info).to(device)
                    lamb = alpha
                    for module, model1_module, model2_module in zip(sampled_model.modules(), self.model_fixed.modules(), self.model.modules()):
                        if 'Conv' in str(type(module)) or 'Linear' in str(type(module)):
                            module.weight.data = ((1-lamb) * model1_module.weight.data + lamb * model2_module.weight.data)
                            if module.bias is not None:
                                module.bias.data = ((1-lamb) * model1_module.bias.data + lamb * model2_module.bias.data)
                        elif 'BatchNorm' in str(type(module)):
                            module.weight.data = ((1-lamb) * model1_module.weight.data + lamb * model2_module.weight.data)
                            module.bias.data = ((1-lamb) * model1_module.bias.data + lamb * model2_module.bias.data)
                            module.running_mean.data = ((1-lamb) * model1_module.running_mean.data + lamb * model2_module.running_mean.data)
                            module.running_var.data = ((1-lamb) * model1_module.running_var.data + lamb * model2_module.running_var.data)
                            module.num_batches_tracked = ((1-lamb) * model1_module.num_batches_tracked.data + lamb * model2_module.num_batches_tracked.data)
                    # for p, p1, p2 in zip(sampled_model.parameters(), self.model_fixed.parameters(), self.model.parameters()):
                    #     p = (1-lamb)*p1 + lamb*p2
                    for task in range(t+1):
                        data = torch.stack([sample[0] for sample in self.memory[task]], dim=0)
                        target = torch.tensor([sample[1] for sample in self.memory[task]])
                        data = data.to(device)
                        target = target.to(device)
                        batch_size = len(self.memory)
                        output = sampled_model(data, task)
                        loss_CE = self.criterion(output,target)
                        sampled_model.zero_grad()
                        loss_CE.backward()
                        self.accum_grad(sampled_model, lamb)

                    sampled_model = networks.ModelFactory.get_model(self.args.model, self.task_info).to(device)
                    lamb = alpha
                    # for p, p1, p2 in zip(sampled_model.parameters(), tmp_model.parameters(), self.model.parameters()):
                    #     p = (1-lamb)*p1 + lamb*p2
                    for module, model1_module, model2_module in zip(sampled_model.modules(), tmp_model.modules(), self.model.modules()):
                        if 'Conv' in str(type(module)) or 'Linear' in str(type(module)):
                            module.weight.data = ((1-lamb) * model1_module.weight.data + lamb * model2_module.weight.data)
                            if module.bias is not None:
                                module.bias.data = ((1-lamb) * model1_module.bias.data + lamb * model2_module.bias.data)
                        elif 'BatchNorm' in str(type(module)):
                            module.weight.data = ((1-lamb) * model1_module.weight.data + lamb * model2_module.weight.data)
                            module.bias.data = ((1-lamb) * model1_module.bias.data + lamb * model2_module.bias.data)
                            module.running_mean.data = ((1-lamb) * model1_module.running_mean.data + lamb * model2_module.running_mean.data)
                            module.running_var.data = ((1-lamb) * model1_module.running_var.data + lamb * model2_module.running_var.data)
                            module.num_batches_tracked = ((1-lamb) * model1_module.num_batches_tracked.data + lamb * model2_module.num_batches_tracked.data)
                    for task in range(t+1):
                        data = torch.stack([sample[0] for sample in self.memory[task]], dim=0)
                        target = torch.tensor([sample[1] for sample in self.memory[task]])
                        data = data.to(device)
                        target = target.to(device)
                        batch_size = len(self.memory)
                        output = sampled_model(data, task)
                        loss_CE = self.criterion(output,target)
                        sampled_model.zero_grad()
                        loss_CE.backward()
                        self.accum_grad(sampled_model, lamb)

                for task in range(t+1):
                    data = torch.stack([sample[0] for sample in self.memory[task]], dim=0)
                    target = torch.tensor([sample[1] for sample in self.memory[task]])
                    data = data.to(device)
                    target = target.to(device)
                    batch_size = len(self.memory)
                    output = self.model(data, task)
                    loss_CE = self.criterion(output,target)
                    loss_CE.backward()

                self.optimizer.step()
                
                for task in range(t+1):
                    train_iterator = torch.utils.data.DataLoader(train_dataset_loaders[task], batch_size=self.args.batch_size, shuffle=True)
                    test_iterator = torch.utils.data.DataLoader(test_dataset_loaders[task], 100, shuffle=False)
                    train_loss,train_acc = self.evaluator.evaluate(self.model, train_iterator, task, self.device)
                    if np.isnan(train_loss):
                        sys.exit()
                    print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
                    test_loss,test_acc=self.evaluator.evaluate(self.model, test_iterator, task, self.device)
                    print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
                    print()





            
    def criterion(self,output,targets):
        # Regularization for all previous tasks
        loss_ce = self.ce(output, targets)
        return loss_ce

    def appendMemory(self, train_loader):
        n_per_task_examples = self.args.memory_size // self.args.tasknum
        n_per_class_examples = n_per_task_examples // self.task_info[0][1]
        start_class = (self.t)*self.task_info[0][1]
        end_class = (self.t+1)*self.task_info[0][1]
        self.memory[self.t] = []

        for class_number in range(start_class, end_class):
            target = (train_loader.labels == class_number)
            class_train_idx = np.random.choice(np.where(target==1)[0], n_per_class_examples, False)

            for i in class_train_idx:
                self.memory[self.t].append(train_loader[i])
    
    def accum_grad(self, from_model, scale):
        for p, p_from in zip(self.model.parameters(), from_model.parameters()):
            p.grad.data += (p_from.grad.data * scale)
