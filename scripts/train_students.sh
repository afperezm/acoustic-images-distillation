#!/usr/bin/env bash

# Experiments with SoundNet5 (pre-trained weights)

python main_s1.py --mode train --model SoundNet5 --exp_name soundnet5_exp1 \
--train_file ./dataset/10_seconds_location_1/lists/training.txt \
--valid_file ./dataset/10_seconds_location_1/lists/validation.txt \
--init_checkpoint ./checkpoints/soundnet/soundnet5_final.t7 \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 5 --number_of_crops 6 --buffer_size 1 \
--log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model SoundNet5 --exp_name soundnet5_exp2 \
--train_file ./dataset/10_seconds_location_2/lists/training.txt \
--valid_file ./dataset/10_seconds_location_2/lists/validation.txt \
--init_checkpoint ./checkpoints/soundnet/soundnet5_final.t7 \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 5 --number_of_crops 6 --buffer_size 1 \
--log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model SoundNet5 --exp_name soundnet5_exp3 \
--train_file ./dataset/10_seconds_location_3/lists/training.txt \
--valid_file ./dataset/10_seconds_location_3/lists/validation.txt \
--init_checkpoint ./checkpoints/soundnet/soundnet5_final.t7 \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 5 --number_of_crops 6 --buffer_size 1 \
--log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model SoundNet5 --exp_name soundnet5_exp7 \
--train_file ./dataset/10_seconds/lists/training.txt \
--valid_file ./dataset/10_seconds/lists/validation.txt \
--init_checkpoint ./checkpoints/soundnet/soundnet5_final.t7 \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 5 --number_of_crops 6 --buffer_size 1 \
--log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

# Experiments with HearNet (min-max norm)

python main_s1.py --mode train --model HearNet --exp_name hearnet_exp1 \
--train_file ./dataset/10_seconds_location_1/lists/training.txt \
--valid_file ./dataset/10_seconds_location_1/lists/validation.txt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 5 --number_of_crops 6 --buffer_size 20 --normalize True \
--log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model HearNet --exp_name hearnet_exp2 \
--train_file ./dataset/10_seconds_location_2/lists/training.txt \
--valid_file ./dataset/10_seconds_location_2/lists/validation.txt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 5 --number_of_crops 6 --buffer_size 20 --normalize True \
--log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model HearNet --exp_name hearnet_exp3 \
--train_file ./dataset/10_seconds_location_3/lists/training.txt \
--valid_file ./dataset/10_seconds_location_3/lists/validation.txt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 5 --number_of_crops 6 --buffer_size 20 --normalize True \
--log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/

python main_s1.py --mode train --model HearNet --exp_name hearnet_exp7 \
--train_file ./dataset/10_seconds/lists/training.txt \
--valid_file ./dataset/10_seconds/lists/validation.txt \
--batch_size 32 --learning_rate 0.0001 --num_epochs 100 \
--sample_length 5 --number_of_crops 6 --buffer_size 20 --normalize True \
--log_dir ./tensorboard/ --checkpoint_dir ./checkpoints/
