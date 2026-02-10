from model import DTLN_model
import os

# use the GPU with idx 0
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# activate this for some reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# path to folder containing the noisy or mixed audio training files
path_to_train_mix = 'datasets/train/noisy'
# path to folder containing the clean/speech files for training
path_to_train_speech = 'datasets/train/clean'
# path to folder containing the noisy or mixed audio validation data
path_to_val_mix = 'datasets/val/noisy'
# path to folder containing the clean audio validation data
path_to_val_speech = 'datasets/val/clean'

# file mapping CSVs for mismatched filenames
train_file_mapping = 'file_mapping_train.csv'
val_file_mapping = 'file_mapping_val.csv'

# name your training run
runName = 'DTLN_model'
# create instance of the DTLN model class
modelTrainer = DTLN_model()
# build the model
modelTrainer.build_DTLN_model()
# compile it with optimizer and cost function for training
modelTrainer.compile_model()
# train the model
modelTrainer.train_model(runName, path_to_train_mix, path_to_train_speech, \
                         path_to_val_mix, path_to_val_speech, 
                         train_file_mapping=train_file_mapping,
                         val_file_mapping=val_file_mapping)