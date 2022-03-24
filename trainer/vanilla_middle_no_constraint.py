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
    def __init__(self, model, args, optimizer, evaluator, taskcla):
        super().__init__(model, args, optimizer, evaluator, taskcla)

        self.lamb=args.lamb


    def train(self, train_loader, test_loader, t, device = None):

        self.device = device
        lr = self.args.lr
        self.setup_training(lr)
        # Do not update self.t
        if t==0:
            #model1 = self.train_from(None, train_loader, test_loader, t, from_pretrained = True)
            model1 = torchvision.models.resnet18(pretrained=False)
            num_ftrs = model1.fc.in_features
            model1.fc = nn.Linear(num_ftrs, self.task_info[0][1])
            model1.to(device)
            model1.load_state_dict(torch.load('./trained_model/_CIFAR100_for_Resnet_from_pretraind_SGD_0_lamb_1.0_lr_0.1_batch_256_epoch_100_task_0.pt'))
            test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
            test_loss,test_acc=self.evaluator.one_task_evaluate(model1, test_iterator, self.device)
            print('From_pretrained MODEL Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()
            #model2 = self.train_from(model1, train_loader, test_loader, t, from_pretrained = False)
            model2 = torchvision.models.resnet18(pretrained=False)
            num_ftrs = model1.fc.in_features
            model2.fc = nn.Linear(num_ftrs, self.task_info[0][1])
            model2.to(device)
            model2.load_state_dict(torch.load('./trained_model/_CIFAR100_for_Resnet_vanilla_basin_constraint_from_model1_SGD_0_lamb_{}_lr_0.1_batch_256_epoch_100_task_0.pt'.format(self.args.lamb)))
            test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
            test_loss,test_acc=self.evaluator.one_task_evaluate(model2, test_iterator, self.device)
            print('From_model1_lamb_{} MODEL Test: loss={:.3f}, acc={:5.1f}% |'.format(self.args.lamb, test_loss,100*test_acc),end='')
            print()
            self.set_model_to_middle(model1, model2)
            self.set_radius(model1, model2)
            test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
            test_loss,test_acc=self.evaluator.evaluate(self.model, test_iterator, t, self.device)
            print('MIDDLE MODEL Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print('Distance from middle to task_0 : {}, radius : {}'.format(self.compute_distance(self.model, model2), self.radius))
            print()
            return
        elif t==1:
            self.update_frozen_model()
            self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.args.batch_size, shuffle=True)
            self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
            self.model.load_state_dict(torch.load('./trained_model/_CIFAR100_for_Resnet_vanilla_basin_constraint_from_pretraind_SGD_0_lamb_{}_lr_0.1_batch_256_epoch_100_task_1.pt'.format(self.args.lamb)))
            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t, self.device)
            num_batch = len(self.train_iterator)
            #print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.evaluate(self.model, self.test_iterator, t, self.device)
            print(' Model_task1 Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()
            return
        else: # update fisher before start training new task
            self.update_frozen_model()
            self.update_fisher()


        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr)
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_lr)
        
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
        save_pt_name = 'from_previous_model'
        log_name = '_{}_{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(self.args.dataset,self.args.trainer,  save_pt_name, self.args.optimizer, self.args.seed,
                                                                           self.args.lamb, self.args.lr, self.args.batch_size, self.args.nepochs)
        torch.save(self.model.state_dict(), './trained_model/' + log_name + '_task_{}.pt'.format(t))
        # self.constrain_to_basin()

    def criterion(self,output,targets):
        """
        Arguments: output (The output logit of self.model), targets (Ground truth label)
        Return: loss function for the classification task

        """
        loss = self.ce
        return loss(output, targets)

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
            self.optimizer = torch.optim.SGD(model_to_train.parameters(), lr=self.current_lr)

        for epoch in range(self.args.nepochs):
            model_to_train.train()
            self.update_lr(epoch, self.args.schedule)
            for samples in tqdm(train_iterator):
                data, target = samples
                data, target = data.to(device), target.to(device)
                batch_size = data.shape[0]

                output = model_to_train(data)
                loss_CE = self.criterion(output,target)
                loss_CE_item = loss_CE.item()
                if not from_pretrained:
                    distance_from_from_model = self.compute_distance(model_to_train, from_model)
                    distance_item = distance_from_from_model.item()
                    loss_CE = loss_CE - (self.lamb * distance_from_from_model)

                


                self.optimizer.zero_grad()
                (loss_CE).backward()
                self.optimizer.step()
        
            if not from_pretrained:
                print(" loss_CE : {} , distance : {} , loss_CE / distance : {}".format(loss_CE_item, distance_item, loss_CE_item / (distance_item+1e-6)))
            train_loss,train_acc = self.evaluator.one_task_evaluate(model_to_train, train_iterator, self.device)
            #num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.one_task_evaluate(model_to_train, test_iterator, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()
        save_pt_name = 'from_pretraind' if from_pretrained else 'from_model1'
        log_name = '_{}_{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(self.args.dataset,self.args.trainer,  save_pt_name, self.args.optimizer, self.args.seed,
                                                                           self.args.lamb, self.args.lr, self.args.batch_size, self.args.nepochs)
        torch.save(model_to_train.state_dict(), './trained_model/' + log_name + '_task_{}.pt'.format(t))
        return model_to_train

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
        self.model.last[0].weight.data = (model1.fc.weight.data + model2.fc.weight.data) / 2.0
        self.model.last[0].bias.data = (model1.fc.bias.data + model2.fc.bias.data) / 2.0

        self.center = copy.deepcopy(self.model)
    
    def set_radius(self, model1, model2):
        self.radius = self.compute_distance(model1, model2).item() / 2.0

    def constrain_to_basin(self):
        distance_from_center = self.compute_distance(self.model, self.center).item()
        with open('distance.txt', 'a') as distance_file:
            distance_file.write('Task {} | Distance from center : {} , Radius : {}\n'.format(self.t, distance_from_center, self.radius))
        if distance_from_center > self.radius:
            print("\nOut of Basin!! Apply Constrain!!\n")
            ratio = distance_from_center / self.radius
            for module, center_module in zip(self.model.modules(), self.center.modules()):
                if 'Conv' in str(type(module)):
                    module.weight.data = ratio * module.weight.data + (1-ratio) * center_module.weight.data
                    if module.bias is not None:
                        module.bias.data = ratio * module.bias.data + (1-ratio) * center_module.bias.data
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data = ratio * module.weight.data + (1-ratio) * center_module.weight.data
                    module.bias.data = ratio * module.bias.data + (1-ratio) * center_module.bias.data
                    module.running_mean.data = ratio * module.running_mean.data + (1-ratio) * center_module.running_mean.data
                    module.running_var.data = ratio * module.running_var.data + (1-ratio) * center_module.running_var.data
                
    def compute_distance(self, model1, model2):
        norm_square_sum = 0
        #for module1, module2 in zip(model1.modules(), model2.modules()):
        #    if 'Conv' in str(type(module1)):
        #        norm_square_sum += torch.norm(module1.weight.data-module2.weight.data)**2
        #        if module1.bias is not None: 
        #            norm_square_sum += torch.norm(module1.bias.data-module2.bias.data)**2
        #    elif 'BatchNorm' in str(type(module1)):
        #        norm_square_sum += torch.norm(module1.weight.data - module2.weight.data)**2
        #        norm_square_sum += torch.norm(module1.bias.data - module2.bias.data)**2
        #        norm_square_sum += torch.norm(module1.running_mean - module2.running_mean)**2
        #        norm_square_sum += torch.norm(module1.running_var - module2.running_var)**2
        for (n1,p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if 'fc' in n1 or 'last' in n1:
                continue
            norm_square_sum += torch.norm(p1-p2)**2
        
        return torch.sqrt(norm_square_sum)
        # param_center_flatten = torch.Tensor().to(self.device)
        # param_new_flatten = torch.Tensor().to(self.device)
        # for param_center, param_new in zip(self.center.paramters(), self.model.paramters()):
        #     param_center_flatten = torch.cat([param_center_flatten, param_center.view(-1)])
        #     param_new_flatten = torch.cat([param_new_flatten, param_new.view(-1)])
        
