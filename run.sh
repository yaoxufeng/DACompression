# A -> W resnet50 train baseline
CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' './Datasets/office31/amazon/images/' './Datasets/office31/webcam/images/' './Checkpoint/AW_train' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --k 1 --batch-size 32 --workers 8 --eval-freq 3 --gpu_ids 0  --method "train" --log_file "./log/AW_train_result.log" --tensorboard_file "./log/AW/train_tensorboard/"
# warning! the default learning rate in config files is for only one GPU and 32imgs/gpu batch_size = 32*1 = 32
# you are supposed to change it according to learning scale rule, eg, lr = 1e-3 for 32imgs/gpu and 4e-3 for 32imgs/4gpu

# A -> W resnet50 train_ada
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' '/users/leo/Datasets/DA/office31/amazon/images/' '/users/leo/Datasets/DA/office31/webcam/images/' './Checkpoint/AW_ada' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --k 2 --batch-size 32 --workers 4 --eval-freq 3 --gpu_ids 0  --method "train_ada" --log_file "./log/AW_train_ada_result.log" --tensorboard_file "./log/AW/train_ada_tensorboard/"

# A -> W resnet50 train_consistency_regulization
#CUDA_VISIBLE_DEVICES=1 python train.py 'Office31' '/users/leo/Datasets/DA/office31/amazon/images/' '/users/leo/Datasets/DA/office31/webcam/images/' './Checkpoint/AW_consistency_regu' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --batch-size 32 --workers 4 --eval-freq 3 --gpu_ids 0 --k 2 --method "consistency_regu" --log_file "./log/AW_train_consistency_result.log" --tensorboard_file "./log/AW/train_consistency_tensorboard/"

# A -> W resnet50 train_mixup
#CUDA_VISIBLE_DEVICES=51 python train.py 'Office31' '/users/leo/Datasets/DA/office31/amazon/images/' '/users/leo/Datasets/DA/office31/webcam/images/' './Checkpoint/AW_mixup' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --batch-size 32 --workers 4 --eval-freq 3 --gpu_ids 0 --method "train_mixup" --log_file "./log/AW_train_mixup_result.log" --tensorboard_file "./log/AW/train_mixup_tensorboard/"

# A -> W resnet50 train_mixatch
#CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py 'Oice31' './Datasets/office31/amazon/images/' './Datasets/office31/webcam/images/' './Checkpoint/AW_mixmatch' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --batch-size 32 --workers 8 --eval-freq 3 --gpu_ids 0 1 2 3 --k 2 --method "train_mixmatch" --log_file "./log/AW_train_mixmatch_result.log" --tensorboard_file "./log/AW/train_mixmatch_tensorboard/"

# A -> D resnet 50 train baseline
#CUDA_VISIBLE_DEVICES=0 python train.py 'Office31' '/users/leo/Datasets/DA/office31/amazon/images/' '/users/leo/Datasets/DA/office31/dslr/images/' './Checkpoint/AD_train' --arch 'resnet50' --num_epoch 50 --optimizer 'SGD' --lr '1e-3' --lr_scheduler 'cosine_decay' --batch-size 64 --workers 4 --eval-freq 3 --gpu_ids 0 --log_file "./log/AD_train_result.log" --tensorboard_file "./log/AD_train_tensorboard/"
#0.784±0.2 (reproduce successfully)

