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
        t_acc = [float(i) for i in line.split(" ")]
        acc.append(sum(t_acc)/(t+1)*100)
    return acc

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
    acc_list = [] 
    for filename in os.listdir(target_dir):
        if filename.endswith(".txt"):
            # loss_dict[filename] = np.load(os.path.join(target_dir, filename))
            if 'ewc_pooling' in filename:
                trainer = 'film_pooling'
            else:
                continue
            lamb = re.search('lamb_(.+?)_', filename).group(1)
            lr = re.search('lr_(.+?)_', filename).group(1)
            acc = get_acc(os.path.join(target_dir, filename))

            acc_list.append((filename, acc))
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
        plt.xticks(np.arange(6))
        plt.xlabel("Tasks")
        plt.ylabel("Avg Acc")

        #for trainer, lamb, lr, acc in acc_list:

        #    #legend = re.search('tilt_(.+?)_', loss_filename)
        #    #if legend:
        #    #    legend = legend.group(1)
        #    #else:
        #    #    sys.exit()

        #    plt.plot(acc,'o-', label='{}, lamb:{}, lr:{}'.format(trainer, lamb, lr))
        for filename, acc in acc_list:
            plt.plot(acc, 'o-', label=filename[10:-4])
        plt.legend()
        plt.savefig(sys.argv[3])

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
