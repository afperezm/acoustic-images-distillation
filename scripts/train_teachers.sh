#!/usr/bin/env bash

# Experiments with DualCamNet + Shared Layers

python main_s1.py --mode train --model DualCamHybridNet --exp_name dualcamnet_exp1 \
--train_file ./dataset/10_seconds_location_1/lists/training.txt \
--valid_file ./dataset/10_seconds_location_1/lists/validation.txt \
--batch_size 32 --learning_rate 0.001 --num_epochs 100 \
--sample_length 1 --number_of_crops 30 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model DualCamHybridNet --exp_name dualcamnet_exp2 \
--train_file ./dataset/10_seconds_location_2/lists/training.txt \
--valid_file ./dataset/10_seconds_location_2/lists/validation.txt \
--batch_size 32 --learning_rate 0.001 --num_epochs 100 \
--sample_length 1 --number_of_crops 30 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model DualCamHybridNet --exp_name dualcamnet_exp3 \
--train_file ./dataset/10_seconds_location_3/lists/training.txt \
--valid_file ./dataset/10_seconds_location_3/lists/validation.txt \
--batch_size 32 --learning_rate 0.001 --num_epochs 100 \
--sample_length 1 --number_of_crops 30 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model DualCamHybridNet --exp_name dualcamnet_exp7 \
--train_file ./dataset/10_seconds/lists/training.txt \
--valid_file ./dataset/10_seconds/lists/validation.txt \
--batch_size 32 --learning_rate 0.001 --num_epochs 100 \
--sample_length 1 --number_of_crops 30 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

# Experiments with ResNet50Model (single video frames)

python main_s1.py --mode train --model ResNet50 --exp_name resnet50_exp1 \
--train_file ./dataset/10_seconds_location_1/lists/training.txt \
--valid_file ./dataset/10_seconds_location_1/lists/validation.txt \
--init_checkpoint ./checkpoints/resnet/resnet_v1_50.ckpt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 1 --number_of_crops 20 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model ResNet50 --exp_name resnet50_exp2 \
--train_file ./dataset/10_seconds_location_2/lists/training.txt \
--valid_file ./dataset/10_seconds_location_2/lists/validation.txt \
--init_checkpoint ./checkpoints/resnet/resnet_v1_50.ckpt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 1 --number_of_crops 20 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model ResNet50 --exp_name resnet50_exp3 \
--train_file ./dataset/10_seconds_location_3/lists/training.txt \
--valid_file ./dataset/10_seconds_location_3/lists/validation.txt \
--init_checkpoint ./checkpoints/resnet/resnet_v1_50.ckpt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 1 --number_of_crops 20 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model ResNet50 --exp_name resnet50_exp7 \
--train_file ./dataset/10_seconds/lists/training.txt \
--valid_file ./dataset/10_seconds/lists/validation.txt \
--init_checkpoint ./checkpoints/resnet/resnet_v1_50.ckpt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 1 --number_of_crops 20 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

# Experiments with ResNet50TemporalModel (random pick five frames + ResNet50 checkpoint)

python main.py --mode train --model TemporalResNet50 --exp_name resnet50_temporal_exp1 \
--train_file ./dataset/10_seconds_location_1/lists/training.txt \
--valid_file ./dataset/10_seconds_location_1/lists/validation.txt \
--init_checkpoint ./checkpoints/resnet/resnet_v1_50.ckpt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 1 --number_of_crops 20 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main.py --mode train --model TemporalResNet50 --exp_name resnet50_temporal_exp2 \
--train_file ./dataset/10_seconds_location_2/lists/training.txt \
--valid_file ./dataset/10_seconds_location_2/lists/validation.txt \
--init_checkpoint ./checkpoints/resnet/resnet_v1_50.ckpt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 1 --number_of_crops 20 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main.py --mode train --model TemporalResNet50 --exp_name resnet50_temporal_exp3 \
--train_file ./dataset/10_seconds_location_3/lists/training.txt \
--valid_file ./dataset/10_seconds_location_3/lists/validation.txt \
--init_checkpoint ./checkpoints/resnet/resnet_v1_50.ckpt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 1 --number_of_crops 20 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main.py --mode train --model TemporalResNet50 --exp_name resnet50_temporal_exp7 \
--train_file ./dataset/10_seconds/lists/training.txt \
--valid_file ./dataset/10_seconds/lists/validation.txt \
--init_checkpoint ./checkpoints/resnet/resnet_v1_50.ckpt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 1 --number_of_crops 20 --log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/
