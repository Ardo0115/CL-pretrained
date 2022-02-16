import copy
from arguments import get_args
args = get_args()
import torch

import sys

class TrainerFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(myModel, args, optimizer, evaluator, task_info):

        if args.trainer == 'ewc' or args.trainer == 'ewc_pooling' or args.trainer=='ewc_wo_fc' or args.trainer=='ewc_resnet34' or args.trainer=='ewc_resnet18_Adam' or args.trainer=='ewc_vgg16':
            import trainer.ewc as trainer
        elif args.trainer == 'mas_resnet18_Adam':
            import trainer.mas as trainer
        elif args.trainer == 'ewc_resnet18_Adam_remember_pretrain':
            import trainer.ewc_remember_pretrain as trainer
        elif args.trainer == 'mas_resnet18_Adam_remember_pretrain':
            import trainer.mas_remember_pretrain as trainer
        elif args.trainer == 'ewc_resnet18_Adam_fixed_bn':
            import trainer.ewc_fixed_bn as trainer
        elif args.trainer == 'ewc_resnet34_SGD' or args.trainer == 'ewc_resnet18_SGD':
            import trainer.ewc_SGD as trainer
        elif args.trainer == 'ewc_w_film_resnet':
            import trainer.ewc_w_film_resnet as trainer
        elif args.trainer == 'vanilla':
            import trainer.vanilla as trainer
        elif args.trainer == 'l2':
            import trainer.l2 as trainer
        elif args.trainer == 'vanilla_new_grad':
            import trainer.vanilla_new_grad as trainer
        elif args.trainer == 'film' or args.trainer == 'film_last' or args.trainer == 'film_pooling' or args.trainer== 'film_wo_fc':
            import trainer.film as trainer
        elif args.trainer == 'film_w_conv':
            import trainer.film_w_conv as trainer
        elif args.trainer == 'film_diag' or args.trainer == 'film_diag_pooling':
            import trainer.film_diag as trainer
        elif args.trainer == 'film_indep_pooling' or args.trainer == 'film_indep':
            import trainer.film_indep as trainer
        elif args.trainer == 'film_remember' or args.trainer == 'film_remember_wo_batchnorm':
            import trainer.film_remember as trainer
        elif args.trainer == 'film_freeze_last':
            import trainer.film_freeze_last as trainer
        elif args.trainer == 'film_remember_freeze' or args.trainer == 'film_remember_freeze_wo_batchnorm':
            import trainer.film_remember_freeze as trainer
        elif args.trainer == 'film_w_conv_all':
            import trainer.film_w_conv_all as trainer
        elif args.trainer == 'film_resnet34_SGD' or args.trainer == 'film_resnet18_SGD':
            import trainer.film_resnet_SGD as trainer
        elif args.trainer == 'film_resnet34_Adam' or args.trainer == 'film_resnet18_Adam':
            import trainer.film_resnet_Adam as trainer
        elif args.trainer == 'film_diag_resnet18_SGD':
            import trainer.film_diag_resnet_SGD as trainer
        elif args.trainer == 'film_diag_resnet18_Adam':
            import trainer.film_diag_resnet_Adam as trainer
        elif args.trainer == 'film_resnet18_Adam_indep':
            import trainer.film_resnet_Adam_indep as trainer
        elif args.trainer == 'film_resnet18_SGD_indep':
            import trainer.film_resnet_SGD_indep as trainer
        elif args.trainer == 'film_w_conv_all_resnet':
            import trainer.film_w_conv_all_resnet as trainer
        elif args.trainer == 'piggyback_resnet18' or args.trainer == 'piggyback_vgg16':
            import trainer.piggyback as trainer
        elif args.trainer == 'film_vgg16':
            import trainer.film_vgg16 as trainer
        elif args.trainer == 'film_indep_vgg16':
            import trainer.film_indep_vgg16 as trainer
        elif args.trainer == 'film_diag_vgg16':
            import trainer.film_diag_vgg16 as trainer
        elif args.trainer == 'only_train_last_resnet18_Adam':
            import trainer.only_train_last as trainer
        else:
            print("Not available trainer")
            sys.exit()

        return trainer.Trainer(myModel, args, optimizer, evaluator, task_info)

class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this.
    '''

    def __init__(self, model, args, optimizer, evaluator, task_info):

        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.evaluator=evaluator
        self.task_info=task_info
        if 'piggyback' not in args.trainer:
            self.model_fixed = copy.deepcopy(self.model)
    #        for param in self.model.parameters():
    #            param.requires_grad = True
            for param in self.model_fixed.parameters():
                param.requires_grad = False
        self.current_lr = args.lr
        self.ce=torch.nn.CrossEntropyLoss()
        if 'piggyback' not in args.trainer:
            self.model_single = copy.deepcopy(self.model)
        self.optimizer_single = None

        self.fisher = dict()
        self.grads = dict()

    # Decrease lr when epoch == schedule[i]
    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]


    def setup_training(self, lr):

        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False
            
