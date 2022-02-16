import torch
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset, trainer, task_info):

        if dataset == 'CIFAR100' or 'CIFAR10':

            import networks.network as net
            if trainer == 'film':
                return net.conv_net_FiLM(task_info)
            elif trainer == 'film_last':
                return net.conv_net_FiLM_last(task_info)
            elif trainer == 'film_pooling' or trainer == 'film_indep_pooling' or trainer == 'film_diag_pooling':
                return net.conv_net_FiLM_pooling(task_info)
            elif trainer == 'ewc':
                return net.conv_net(task_info)
            elif trainer == 'ewc_pooling':
                return net.conv_net_pooling(task_info)
            elif trainer == 'film_wo_fc' or trainer == 'film_w_conv' or trainer == 'film_diag' or trainer == 'film_freeze_last' or trainer == 'film_w_conv_all' or trainer == 'film_indep':
                return net.conv_net_FiLM_wo_fc(task_info)
            elif trainer == 'ewc_wo_fc':
                return net.conv_net_wo_fc(task_info)
            elif trainer == 'film_remember' or trainer == 'film_remember_freeze':
                return net.conv_net_FiLM_remember(task_info)
            elif trainer == 'film_remember_wo_batchnorm' or trainer == 'film_remember_freeze_wo_batchnorm':
                return net.conv_net_FiLM_remember_wo_batchnorm(task_info)
            elif trainer == 'ewc_resnet34' or trainer == 'ewc_resnet34_SGD':
                return net.resnet34(task_info)
            elif trainer == 'film_resnet34_Adam' or trainer == 'film_resnet34_SGD':
                return net.resnet34_FiLM(task_info)
            elif trainer == 'ewc_resnet18_Adam' or trainer == 'ewc_resnet18_SGD' or trainer == 'ewc_resnet18_Adam_remember_pretrain' or trainer == 'ewc_resnet18_Adam_fixed_bn' or trainer == 'mas_resnet18_Adam' or trainer == 'mas_resnet18_Adam_remember_pretrain':
                return net.resnet18(task_info)
            elif trainer == 'film_resnet18_Adam' or trainer == 'film_resnet18_SGD' or trainer == 'film_diag_resnet18_SGD' or trainer == 'film_diag_resnet18_Adam' or trainer == 'film_resnet18_Adam_indep' or trainer == 'film_resnet18_SGD_indep' or trainer == 'film_w_conv_all_resnet' or trainer == 'ewc_w_film_resnet' or trainer == 'only_train_last_resnet18_Adam':
                return net.resnet18_FiLM(task_info)
            elif trainer == 'piggyback_resnet18':
                return net.resnet18_piggyback(task_info)
            elif trainer == 'ewc_vgg16':
                return net.vgg16_original(task_info)
            elif trainer == 'film_vgg16' or trainer == 'film_indep_vgg16' or trainer == 'film_diag_vgg16':
                return net.vgg16_film(task_info)
            elif trainer == 'piggyback_vgg16':
                return net.vgg16_piggyback(task_info)
            else:
                print("Invalid trainer")
                sys.exit()

        elif dataset == 'MNIST':

            import networks.network as net
            return net.MLP(task_info)
