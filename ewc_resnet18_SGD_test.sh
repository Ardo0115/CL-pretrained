lamb="1"
lamb_limit="10000000"
lamb_increment="10"

while [ "$(bc <<< "$lamb <= $lamb_limit")" == "1"  ]; do
    CUDA_VISIBLE_DEVICES=3 python main.py --trainer ewc --model resnet18 --optimizer SGD --dataset CIFAR100_for_Resnet --nepochs 60 --lr 0.1 --lamb $lamb --batch-size 256 --seed 0 --tasknum 20 --schedule 40 80
    lamb=$(bc <<< "$lamb*$lamb_increment")
done

#lamb="10"
#lamb_limit="90"
#lamb_increment="10"
#
#while [ "$(bc <<< "$lamb <= $lamb_limit")" == "1"  ]; do
#    CUDA_VISIBLE_DEVICES=1 python main.py --trainer ewc_resnet18_Adam --dataset CIFAR100 --nepochs 60 --lr 0.001 --lamb $lamb --batch-size 256 --seed 0 --tasknum 20
#    lamb=$(bc <<< "$lamb+$lamb_increment")
#done
