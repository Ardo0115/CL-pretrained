import sys, os, time
import numpy as np

import pickle

from numpy.core.defchararray import array
from arguments import get_args
import random

import matplotlib.pyplot as plt
import re

def get_acc(filename):
    target_file = open(filename, "r")
    content = target_file.readlines()
    acc = []
    for t, line in enumerate(content):
        t_acc = [float(i) for i in line.split(" ")]
        acc.append(sum(t_acc)/(t+1)*100)
    return acc

def get_bwt(filename):
    target_file = open(filename, "r")
    content = target_file.readlines()
    last_acc = [float(i) for i in content[-1].split(" ")]
    for t, line in enumerate(content):
        if t == len(content)-1:
            break
        t_acc = [float(i) for i in line.split(" ")]
        R_ii = t_acc[t]
        last_acc[t] = last_acc[t] - R_ii
    return sum(last_acc[0:-1])/float(len(last_acc[0:-1]))
def main():
    if len(sys.argv) != 4:
        print('Argument 1 : target directory to plot')
        print('Argument 2 : Plot in One figure - 1, Plot in multiple figure - 2')
        print('Argument 3 : Name for ouput figure')
        sys.exit()
    print('Load data...')
    loss_dict = dict()
    target_dir = sys.argv[1]
    plot_type = int(sys.argv[2])
    acc_list = {} 
    for filename in os.listdir(target_dir):
        if filename.endswith(".txt"):
            # loss_dict[filename] = np.load(os.path.join(target_dir, filename))
            #if 'film_0' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'film'
            #elif 'film_last' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'film_last'
            #if 'film_pooling' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'film_pooling'
            #elif 'ewc_0' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'ewc'
            #elif 'ewc_pooling' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'ewc_pooling'

            #if 'ewc_wo_fc_0' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'ewc_wo_fc'
            #elif 'film_wo_fc_0' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'film_wo_fc'
            #elif 'film_w_conv_0' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'film_w_conv'
            #elif 'film_diag_0' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'film_diag'
            #elif 'film_remember_0' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'film_remember'
            #elif 'film_remember_wo_batchnorm_0' in filename:
            #    trainer = 'film_remember_wo_batchnorm'
            #elif 'film_remember_freeze_wo_batchnorm' in filename:
            #    trainer = 'film_remember_freeze_wo_batchnorm'
            #elif 'film_freeze_last_0' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'film_freeze_last'
            #elif 'film_remember_freeze_0' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'film_remember_freeze'
            #elif 'film_w_conv_all_0' in filename and 'lr_0.001' in filename and 'epoch_60' in filename:
            #    trainer = 'film_w_conv_all'
            #if 'ewc_resnet34_0' in filename:
            #    trainer = 'ewc_resnet34_Adam'
            #elif 'ewc_resnet34_SGD_0' in filename:
            #    trainer = 'ewc_resnet34_SGD'
            #elif 'film_resnet34_Adam_0' in filename:
            #    trainer = 'film_resnet34_Adam'
            #elif 'film_resnet34_SGD_0' in filename:
            #    trainer = 'film_resnet34_SGD'

            #if 'ewc_resnet18_SGD_0' in filename:
            #    trainer = 'ewc_resnet18_SGD'
            #elif 'film_resnet18_SGD_0' in filename:
            #    trainer = 'film_resnet18_SGD'
            #elif 'film_diag_resnet18_SGD_0' in filename:
            #    trainer = 'film_diag_resnet18_SGD'
            #if 'CIFAR100' in filename and 'knowledge' in filename and 'epoch_60' in filename and 'remember' not in filename:
            #    trainer = 'mas'
            if 'CIFAR100_for_Resnet' in filename and 'epoch_60' in filename and 'Adam' in filename and 'ewc' in filename:
                trainer = 'ewc_Adam'
            elif 'CIFAR100_for_Resnet' in filename and 'epoch_100' in filename and 'SGD' in filename and 'ewc' in filename:
                trainer = 'ewc_SGD'
            # elif 'CIFAR100_for_Resnet' in filename and 'epoch_60' in filename and 'Adam' in filename and 'interpolate_pretrain' in filename:
            #     trainer = 'interpolate_pretrain_Adam'
            elif 'CIFAR100_for_Resnet' in filename and 'epoch_100' in filename and 'SGD' in filename and 'interpolate_pretrain' in filename:
                trainer = 'interpolate_pretrain_SGD'
            # elif 'CIFAR100_for_Resnet' in filename and 'epoch_100' in filename:
            #     trainer = filename
           
            
            # elif 'CIFAR100' in filename and 'knowledge_0.9' in filename and 'epoch_60' in filename:
            #     trainer = filename

            # if 'ewc_resnet18_Adam_0_knowledge_0.1' in filename:
            #     trainer = 'ewc_resnet18_Adam_knowledge_0.1'
            # elif 'ewc_resnet18_Adam_remember_pretrain_0_knowledge_0.1' in filename:
            #     trainer = 'ewc_resnet18_Adam_remember_pretrain_knowledge_0.1'
            # elif 'ewc_resnet18_Adam_0_knowledge_0.2' in filename:
            #     trainer = 'ewc_resnet18_Adam_knowledge_0.2'
            # elif 'ewc_resnet18_Adam_remember_pretrain_0_knowledge_0.2' in filename:
            #     trainer = 'ewc_resnet18_Adam_remember_pretrain_knowledge_0.2'


            # elif 'ewc_resnet18_Adam_fixed_bn_0' in filename:
            #     trainer = 'ewc_resnet18_Adam_fixed_bn'
            #if 'CIFAR100_film_resnet18_Adam_0' in filename:
            #    trainer = 'film_resnet18_Adam'
            #elif 'film_diag_resnet18_Adam_0' in filename:
            #    trainer = 'film_diag_resnet18_Adam'
            #elif 'film_resnet18_Adam_indep_0' in filename:
            #    trainer = 'film_resnet18_Adam_indep'
            #elif 'only_train_last_resnet18_Adam_0' in filename:
            #    trainer = 'only_train_last'
            #elif 'film_resnet18_SGD_indep_0':
            #    trainer = 'film_resnet18_SGD_indep'

            #if 'ewc' in filename:
            #    trainer = 'ewc'
            #if 'CIFAR100_film_vgg16_0' in filename:
            #    trainer = 'film_vgg16'
            #elif 'film_diag_vgg16_0' in filename:
            #    trainer = 'film_diag_vgg16'
            #elif 'film_indep_vgg16_0' in filename:
            #    trainer = 'film_indep_vgg16'
            else:
                continue

            '''result_data_cmp_fixed'''

            ##if 'moving_avg_CIFAR100_film_resnet18_Adam_0' in filename:
            ##    trainer = 'moving_avg_film_resnet18_Adam'
            #if 'fixed_bn_CIFAR100_film_resnet18_Adam_0' in filename:
            #    trainer = 'fixed_bn_film_resnet18_Adam'
            ##elif 'moving_avg_CIFAR100_film_diag_resnet18_Adam_0' in filename:
            ##    trainer = 'moving_avg_film_diag_resnet18_Adam'
            #elif 'fixed_bn_CIFAR100_film_diag_resnet18_Adam_0' in filename:
            #    trainer = 'fixed_bn_film_diag_resnet18_Adam'
            ##elif 'moving_avg_CIFAR100_film_resnet18_Adam_indep_0' in filename:
            ##    trainer = 'moving_avg_film_resnet18_Adam_indep'
            #elif 'fixed_bn_CIFAR100_film_resnet18_Adam_indep_0' in filename:
            #    trainer = 'fixed_bn_film_resnet18_Adam_indep'
            ##elif 'moving_avg_CIFAR100_piggyback_resnet18_0' in filename:
            ##    trainer = 'moving_avg_piggyback'
            #elif 'fixed_bn_CIFAR100_piggyback_resnet18_0' in filename:
            #    trainer = 'fixed_bn_piggyback'
            ##elif 'per_task_bn_CIFAR100_film_resnet18_Adam_0' in filename:
            ##    trainer = 'per_task_bn_film'
            ##elif 'per_task_bn_CIFAR100_piggyback_resnet18_0' in filename:
            ##    trainer = 'per_task_bn_piggyback'

            #else:
            #    continue
            lamb = re.search('lamb_(.+?)_', filename).group(1)
            lr = re.search('lr_(.+?)_', filename).group(1)
            acc = get_acc(os.path.join(target_dir, filename))
            if trainer not in acc_list.keys():
                acc_list[trainer] = (filename, acc)
            else:
                if acc_list[trainer][1][-1] < acc[-1]:
                    acc_list[trainer] = (filename, acc)

            #if acc[-1] > 65.5:
            #    acc_list.append((filename, acc))
            #if len(acc_list) == 0:
            #    acc_list.append((trainer, lamb, lr, acc))
            #else:
            #    if acc[-1] > acc_list[-1][-1][-1]:
            #        del(acc_list[-1])
            #        acc_list.append((trainer, lamb, lr, acc))
            #    else:
            #        continue
        else:
            continue

    #acc_list.sort(key=lambda x:(x[0], float(x[1]), float(x[2])))
    #acc_list.sort(key=lambda x:x[1])
# Plot in one Figure
    if plot_type == 1:
        title = "Average ACC"
        plt.figure(figsize=(10,5))
        plt.title(title)
        #plt.xticks(np.arange(1,6), labels=['2', '3', '4', '5'])
        plt.xticks(np.arange(20))
        plt.xlabel("Tasks")
        plt.ylabel("Avg Acc")

        #for trainer, lamb, lr, acc in acc_list:

        #    #legend = re.search('tilt_(.+?)_', loss_filename)
        #    #if legend:
        #    #    legend = legend.group(1)
        #    #else:
        #    #    sys.exit()

        #    plt.plot(acc,'o-', label='{}, lamb:{}, lr:{}'.format(trainer, lamb, lr))
        acc_list_values = list(acc_list.values())
        acc_list_values.sort(key=lambda x:x[1][-1], reverse=True)
        knowledge_list = []
        for filename, acc in acc_list_values:
            knowledge_list.append((filename, round(acc[-1],2)))
            plt.plot(acc, 'o-', label=filename[4:-10])
            plt.text(len(acc)-1, acc[-1], "acc: "+str(acc[-1]))

        
        plt.legend()
        plt.grid()
        plt.savefig(sys.argv[3])
        # print("knowledge diff : {}".format(sum(np.array(knowledge_list[0])-np.array(knowledge_list[1]))))


        
        title = "knowledge diff"
        plt.figure(figsize=(10,5))
        plt.title(title)
        plt.xlabel("Knowledge")
        plt.ylabel("Avg Accuracy Diff")
        ticklabel=list((round(i,1)) for i in np.arange(0.1,1.1,0.1))
        plt.xticks(list(np.arange(0, 10,1)),ticklabel)

        knowledge_x = []
        knowledge_diff_list = []
        for i in range(1, 11):
            knowledge_diff = 0
            for filename, acc in knowledge_list:
                if str(round(i*0.1,2)) in filename:
                    if knowledge_diff == 0:
                        knowledge_diff = acc
                    else:
                        knowledge_diff -= acc
            knowledge_diff_list.append(round(knowledge_diff,2))
        plt.plot(knowledge_diff_list, 'o-')
        for i in range(10):
            plt.text(i, knowledge_diff_list[i], str(knowledge_diff_list[i]))
        plt.grid()

        plt.savefig("knowledge_diff_"+sys.argv[3])

        
        title = "BWT"
        plt.figure(figsize=(10,5))
        plt.title(title)
        trainer_list = []
        bwt_list = []
        for trainer, (filename, _) in acc_list.items():
            trainer_list.append(trainer)
            bwt_list.append(get_bwt(os.path.join(target_dir, filename)))
        x = np.arange(len(trainer_list))
        for i in x:
            plt.bar(i,bwt_list[i])
            #plt.text(i,bwt_list[i], str(round(bwt_list[i],4)))
            plt.text(i,bwt_list[i], trainer_list[i])
        #plt.bar(x, bwt_list)
        #plt.xticks(x, trainer_list, rotation=45)
        plt.gcf().subplots_adjust(bottom=0.30)
        plt.savefig('bar_'+sys.argv[3])
# Plot in Multiple Figure
    elif plot_type == 2:
        fig = plt.figure()
        for tilt, loss_arr in loss_list:
            #legend = re.search('tilt_(.+?)_', loss_filename)
            #if legend:
            #    legend = legend.group(1)
            #else:
            #    sys.exit()
            ax = fig.add_subplot(4,3,int(float(tilt))+1)
            ax.plot(loss_arr, label='tilt = {}/10.0*pi'.format(tilt))
            ax.legend()
        plt.savefig(sys.argv[3], dpi=300)
    else:
        print('Plot type must be 1 or 2')
        sys.exit()






if __name__=="__main__":
    main()
