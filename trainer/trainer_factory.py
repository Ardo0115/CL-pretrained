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

        if args.trainer == 'ewc':
            import trainer.ewc as trainer
        elif args.trainer == 'ewc_from_random':
            import trainer.ewc_from_random as trainer
        elif args.trainer == 'interpolate_pretrain':
            import trainer.interpolate_pretrain as trainer
        elif args.trainer == 'interpolate_random':
            import trainer.interpolate_random as trainer
        elif args.trainer == 'interpolate_pretrain_fix_var':
            import trainer.interpolate_pretrain_fix_var as trainer
        elif args.trainer == 'vanilla_basin_constraint':
            import trainer.vanilla_basin_constraint as trainer
        elif args.trainer == 'vanilla_basin_immediate_constraint':
            import trainer.vanilla_basin_immediate_constraint as trainer
        elif args.trainer == 'vanilla_middle_from_center':
            import trainer.vanilla_middle_from_center as trainer
        elif args.trainer == 'vanilla_middle_no_constraint':
            import trainer.vanilla_middle_no_constraint as trainer
        elif args.trainer == 'vanilla_model1_no_constraint':
            import trainer.vanilla_model1_no_constraint as trainer
        elif args.trainer == 'joint_multihead':
            import trainer.joint_multihead as trainer
        elif args.trainer == 'vanilla_model1_no_constraint_from_pretrain':
            import trainer.vanilla_model1_no_constraint_from_pretrain as trainer
        elif args.trainer == 'vanilla':
            import trainer.vanilla as trainer
        elif args.trainer == 'vanilla_from_task1':
            import trainer.vanilla_from_task1 as trainer
        elif args.trainer == 'vanilla_only_classifier':
            import trainer.vanilla_only_classifier as trainer
        elif args.trainer == 'vanilla_only_classifier_evalmode':
            import trainer.vanilla_only_classifier_evalmode as trainer
        elif args.trainer == 'interpolate_pretrain_fix_bn':
            import trainer.interpolate_pretrain_fix_bn as trainer
        elif args.trainer == 'mc_sgd':
            import trainer.mc_sgd as trainer
        elif args.trainer == 'mc_sgd_from_pretrain':
            import trainer.mc_sgd_from_pretrain as trainer
        elif args.trainer == 'averaging_from_previous':
            import trainer.averaging_from_previous as trainer
        elif args.trainer == 'half_interpolate_pretrain':
            import trainer.half_interpolate_pretrain as trainer
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
            
