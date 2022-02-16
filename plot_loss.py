import sys, os, time
import numpy as np

import pickle
from arguments import get_args
import random

import matplotlib.pyplot as plt
import re

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
    loss_list = []
    for filename in os.listdir(target_dir):
        if filename.endswith(".npy"):
            # loss_dict[filename] = np.load(os.path.join(target_dir, filename))
            tilt = re.search('tilt_(.+?)_', filename).group(1)
            loss_list.append((tilt, np.load(os.path.join(target_dir, filename))))
        else:
            continue

    loss_list.sort(key=lambda x:float(x[0]))
    loss_list = loss_list[:int(len(loss_list)/2)+1]
# Plot in one Figure
    if plot_type == 1:
        title = "Training Loss for for {}~{}".format('0(rad)', 'pi/2(rad)')
        plt.figure(figsize=(10,5))
        plt.title(title)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        for tilt, loss_arr in loss_list:

            #legend = re.search('tilt_(.+?)_', loss_filename)
            #if legend:
            #    legend = legend.group(1)
            #else:
            #    sys.exit()

            plt.plot(loss_arr, label='tilt='+tilt)
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
