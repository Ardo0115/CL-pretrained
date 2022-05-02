import argparse
from re import M


def get_args():
    parser = argparse.ArgumentParser(description='Continual Learning')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    # CUB: 0.005
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1. Note that lr is decayed by args.gamma parameter args.schedule ')
    parser.add_argument('--decay', type=float, default=0, help='Weight decay (L2 penalty).')
    parser.add_argument('--lamb', type=float, default=0, help='Lambda for ewc')
    parser.add_argument('--lamb2', type=float, default=0, help='Lambda for film_w_conv')
    parser.add_argument('--schedule', type=int, nargs='+', default=[20, 40],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds values to be used; seed introduces randomness by changing order of classes')
    parser.add_argument('--nepochs', type=int, default=60, help='Number of epochs for each increment')
    parser.add_argument('--tasknum', default=20, type=int, help='(default=%(default)s)')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--dataset', default='CIFAR100', type=str,
                        choices=['CIFAR100', 'MNIST', 'CIFAR10', 'CIFAR100_for_Resnet'],
                        help='(default=%(default)s)')

    parser.add_argument('--trainer', default='ewc', type=str,
                        choices=['vanilla', 'vanilla_from_task1', 'vanilla_only_classifier', 'vanilla_only_classifier_evalmode', 'ewc','interpolate_pretrain', 'interpolate_random', 'interpolate_pretrain_fix_var', 'vanilla_basin_constraint', 'vanilla_basin_immediate_constraint', 'vanilla_middle_from_center', 'vanilla_middle_no_constraint', 'vanilla_model1_no_constraint', 'ewc_from_random', 'joint_multihead', 'vanilla_model1_no_constraint_from_pretrain', 'interpolate_pretrain_fix_bn'],
                        help='(default=%(default)s)')
    parser.add_argument('--model', default='resnet18',type=str,
                        choices=['resnet18','MLP', 'resnet18_LMC', 'resnet18_small'],
                        help='Model to use')
    parser.add_argument('--optimizer', default='Adam', type=str,
                        choices=['Adam','SGD'],
                        help='Optimizer to use')
    parser.add_argument('--grad-tilt', default='0.0', type=float,
                        help='Choose how much to tilt the gradient from [0.0, 10.0], 0.0:no tilting, 10.0:orthogonal')
    parser.add_argument('--knowledge-ratio', default='1.0', type=float,
                        help='Choose how much to use for CIFAR10 dataset')

    args = parser.parse_args()
    return args
