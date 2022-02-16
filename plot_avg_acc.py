import sys, os, time
import numpy as np

import pickle
from arguments import get_args
import random

import matplotlib.pyplot as plt
import re

def get_acc(filename):
    target_file = open(filename, "r")
    content = target_file.readlines()
    acc = []
    for t, line in enumerate(content):
        if t == 0:
            continue
        else:
            t_acc = [float(i) for i in line.split(" ")]
            acc.append(sum(t_acc[1:])/(t)*100)
    return acc

#def get_acc(filename):
#    target_file = open(filename, "r")
#    content = target_file.readlines()
#    acc = []
#    for t, line in enumerate(content):
#        if t == 0 or t ==1:
#            continue
#        else:
#            t_acc = [float(i) for i in line.split(" ")]
#            acc.append(sum(t_acc[2:])/(t-1)*100)
#    return acc
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
    return sum(last_acc[1:-1])/float(len(last_acc[1:-1]))
#def get_bwt(filename):
#    target_file = open(filename, "r")
#    content = target_file.readlines()
#    last_acc = [float(i) for i in content[-1].split(" ")]
#    for t, line in enumerate(content):
#        if t == len(content)-1:
#            break
#        t_acc = [float(i) for i in line.split(" ")]
#        R_ii = t_acc[t]
#        last_acc[t] = last_acc[t] - R_ii
#    return sum(last_acc[2:-1])/float(len(last_acc[2:-1]))


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
            #elif 'film_diag_pooling_0' in filename:
            #    trainer = 'film_diag_pooling'
            #elif 'film_indep_pooling_0' in filename:
            #    trainer = 'film_indep_pooling'

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
            #if 'ewc_resnet18_Adam_0' in filename:
            #    trainer = 'ewc_resnet18_Adam'
            #elif 'film_resnet18_Adam_0' in filename:
            #    trainer = 'film_resnet18_Adam'
            #elif 'film_diag_resnet18_Adam_0' in filename:
            #    trainer = 'film_diag_resnet18_Adam'
            #elif 'film_resnet18_Adam_indep_0' in filename:
            #    trainer = 'film_resnet18_Adam_indep'
            #elif 'film_resnet18_SGD_indep_0' in filename:
            #    trainer = 'film_resnet18_SGD_indep'
            #elif 'ewc_w_film_resnet_0' in filename:
            #    trainer = 'ewc_w_film_resnet'
            #elif 'film_w_conv_all_resnet_0' in filename:
            #    trainer = 'film_w_conv_all_resnet'

            #if 'ewc' in filename:
            #    trainer = 'ewc'
            if 'ewc_resnet18_Adam_0' in filename:
                trainer = 'ewc_resnet18_Adam'
            elif 'ewc_resnet18_Adam_remember_pretrain_0' in filename:
                trainer = 'ewc_resnet18_Adam_remember_pretrain'
            else:
                continue
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
        plt.xticks(np.arange(1,6), labels=['2', '3', '4', '5'])
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
        for filename, acc in acc_list_values:
            plt.plot(acc, 'o-', label=filename[10:-4])
            plt.text(len(acc)-1, acc[-1], "acc: "+str(acc[-1]))
        plt.legend()
        plt.grid()
        plt.savefig(sys.argv[3])

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
            plt.text(i,bwt_list[i], str(round(bwt_list[i],4)))
        #plt.bar(x, bwt_list)
        plt.xticks(x, trainer_list, rotation=45)
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
