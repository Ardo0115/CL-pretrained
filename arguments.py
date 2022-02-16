import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual Learning')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    # CUB: 0.005
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1. Note that lr is decayed by args.gamma parameter args.schedule ')
    parser.add_argument('--decay', type=float, default=0, help='Weight decay (L2 penalty).')
    parser.add_argument('--lamb', type=float, default=1, help='Lambda for ewc')
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
                        choices=['CIFAR100', 'MNIST', 'CIFAR10'],
                        help='(default=%(default)s)')

    parser.add_argument('--trainer', default='ewc', type=str,
                        choices=['ewc', 'ewc_pooling','ewc_wo_fc',  'l2', 'vanilla', 'vanilla_new_grad', 'film', 'film_last', 'film_pooling', 'film_wo_fc', 'film_w_conv', 'film_diag','film_diag_pooling', 'film_indep_pooling','film_indep', 'film_remember', 'film_freeze_last', 'film_remember_freeze', 'film_w_conv_all', 'ewc_resnet34', 'ewc_resnet34_SGD', 'film_resnet34_SGD', 'film_resnet34_Adam', 'ewc_resnet18_Adam', 'ewc_resnet18_SGD', 'film_resnet18_SGD', 'film_resnet18_Adam', 'film_diag_resnet18_SGD','film_diag_resnet18_Adam', 'film_remember_wo_batchnorm', 'film_remember_freeze_wo_batchnorm','film_resnet18_Adam_indep', 'film_resnet18_SGD_indep', 'film_w_conv_all_resnet', 'ewc_w_film_resnet', 'piggyback_resnet18', 'ewc_vgg16', 'film_vgg16', 'film_indep_vgg16', 'film_diag_vgg16', 'piggyback_vgg16', 'only_train_last_resnet18_Adam', 'ewc_resnet18_Adam_remember_pretrain', 'ewc_resnet18_Adam_fixed_bn', 'mas_resnet18_Adam', 'mas_resnet18_Adam_remember_pretrain'],
                        help='(default=%(default)s)')
    parser.add_argument('--grad-tilt', default='0.0', type=float,
                        help='Choose how much to tilt the gradient from [0.0, 10.0], 0.0:no tilting, 10.0:orthogonal')
    parser.add_argument('--knowledge-ratio', default='1.0', type=float,
                        help='Choose how much to use for CIFAR10 dataset')

    args = parser.parse_args()
    return args
