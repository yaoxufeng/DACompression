# A -> W resnet50 train baseline
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' './Datasets/office31/amazon/images/' './Datasets/office31/webcam/images/' './Checkpoint/AW_train' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --k 1 --batch-size 32 --workers 8 --eval-freq 3 --gpu_ids 0  --method "train" --log_file "./log/AW_train_result.log" --tensorboard_file "./log/AW/train_tensorboard/"
# warning! the default learning rate in config files is for only one GPU and 32imgs/gpu batch_size = 32*1 = 32
# you are supposed to change it according to learning scale rule, eg, lr = 1e-3 for 32imgs/gpu and 4e-3 for 32imgs/4gpu
# result 0.7962 ± 0.05 (it's the last epoch result and we can not guarantee the result because the data augmentation method is random and we are pretty sure that even a random seed wii
# cause a big change in performance in small dataset, however, just feel free to test best parameters!)

# A -> D resnet50 train baseline
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' './Datasets/office31/amazon/images/' './Datasets/office31/dslr/images/' './Checkpoint/AD_train' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --k 1 --batch-size 32 --workers 8 --eval-freq 3 --gpu_ids 0  --method "train" --log_file "./log/AD_train_result.log" --tensorboard_file "./log/AD/train_tensorboard/"
# result 0.8433

# W -> A resnet50 train_baseline
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' './Datasets/office31/webcam/images/' './Datasets/office31/amazon/images/' './Checkpoint/WA_train' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --k 1 --batch-size 32 --workers 8 --eval-freq 3 --gpu_ids 0  --method "train" --log_file "./log/WA_train_result.log" --tensorboard_file "./log/WA/train_tensorboard/"
# result 0.6201

# W -> D resnet50 train_baseline
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' './Datasets/office31/webcam/images/' './Datasets/office31/dslr/images/' './Checkpoint/WD_train' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --k 1 --batch-size 32 --workers 8 --eval-freq 3 --gpu_ids 0  --method "train" --log_file "./log/WD_train_result.log" --tensorboard_file "./log/WD/train_tensorboard/"
# result 0.9979

# D -> A resnet50 train_baseline
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' './Datasets/office31/dslr/images/' './Datasets/office31/amazon/images/' './Checkpoint/WD_train' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --k 1 --batch-size 32 --workers 8 --eval-freq 3 --gpu_ids 0  --method "train" --log_file "./log/WD_train_result.log" --tensorboard_file "./log/WD/train_tensorboard/"
# result 0.6636

# D -> W resnet50 train_baseline
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' './Datasets/office31/dslr/images/' './Datasets/office31/webcam/images/' './Checkpoint/WD_train' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --k 1 --batch-size 32 --workers 8 --eval-freq 3 --gpu_ids 0  --method "train" --log_file "./log/WD_train_result.log" --tensorboard_file "./log/WD/train_tensorboard/"
# result 0.9735

# A -> W resnet50 train_center
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' '/users/leo/Datasets/DA/office31/amazon/images/' '/users/leo/Datasets/DA/office31/webcam/images/' './Checkpoint/AW_center' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --k 1 --batch-size 32 --workers 4 --eval-freq 3 --gpu_ids 0  --method "train_center" --log_file "./log/AW_train_center_result.log" --tensorboard_file "./log/AW/train_center_tensorboard/"
# result 0.7597

# A -> D resnet50 train_center
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' './Datasets/DA/office31/amazon/images/' './Datasets/DA/office31/dslr/images/' './Checkpoint/AD_center' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --k 1 --batch-size 32 --workers 4 --eval-freq 3 --gpu_ids 0  --method "train_center" --log_file "./log/AD_train_center_result.log" --tensorboard_file "./log/AD/train_center_tensorboard/"
# result 0.7349

# A -> W resnet50 train_mixup
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' '/users/leo/Datasets/DA/office31/amazon/images/' '/users/leo/Datasets/DA/office31/webcam/images/' './Checkpoint/AW_mixup' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --batch-size 32 --workers 4 --eval-freq 3 --gpu_ids 0 --k 1 --method "train_mixup" --log_file "./log/AW_mixup_result.log" --tensorboard_file "./log/AW/train_mixup_tensorboard/"

# A -> D resnet50 train_mixup
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' '/users/leo/Datasets/DA/office31/amazon/images/' '/users/leo/Datasets/DA/office31/dslr/images/' './Checkpoint/AD_mixup' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --batch-size 32 --workers 4 --eval-freq 3 --gpu_ids 0 --k 1 --method "train_mixup" --log_file "./log/AD_mixup_result.log" --tensorboard_file "./log/AD/train_mixup_tensorboard/"

# A -> W resnet50 train_mixmatch
CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py 'Office31' '/users/leo/Datasets/DA/office31/amazon/images/' '/users/leo/Datasets/DA/office31/webcam/images/' './Checkpoint/AW_mixup' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --batch-size 32 --workers 4 --eval-freq 3 --gpu_ids 0 1 2 3 --k 2 --method "train_mixmatch" --log_file "./log/AW_mixmatch_result.log" --tensorboard_file "./log/AW/train_mixmatch_tensorboard/"
