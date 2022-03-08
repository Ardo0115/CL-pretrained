CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --trainer vanilla_basin_constraint --model resnet18 --optimizer SGD --lamb 1 --dataset CIFAR100_for_Resnet --nepochs 100 --lr 0.1 --batch-size 256 --seed 0 --tasknum 20 --schedule 40 80

