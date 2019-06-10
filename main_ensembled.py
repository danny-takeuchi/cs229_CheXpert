#loading libraries

import os
import numpy as np
import sys
import cv2
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as tfunc
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
import sklearn.metrics as metrics
import random

from datasetRay import CheXpertDataSet
from testMultiView import CheXpertTrainer
import models as mod
from heatmap import HeatmapGenerator



# Neural network parameters:
nnIsTrained = False                 #pre-trained using ImageNet
nnClassCount = 14                   #dimension of the output

# Training settings: batch size, maximum number of epochs
# ["DenseNet121","Vgg16","Vgg19"]
modelName = "DenseNet121"
policy = "ones"
trBatchSize = 64
trMaxEpoch = 3
action = "test" # train or test
paCheckpoint = "checkpoints/specificTrainModels/pa/m-epoch1-Vgg19-ones-08062019-133446.pth.tar"
apCheckpoint = "checkpoints/specificTrainModels/ap/m-epoch1-Vgg19-ones-08062019-235513.pth.tar"
latCheckpoint = "checkpoints/specificTrainModels/lateral/m-epoch2-Vgg19-ones-08062019-080125.pth.tar"

# onesModeltoTest = "checkpoints/mixedTrainModels/Vgg16-ones/m-epoch1-Vgg16-ones-07062019-052816.pth.tar"
# zerosModeltoTest = "m-epoch2-Vgg19-zeros-260019-135938.pth.tar"

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = (320, 320)
imgtransCrop = 224

# Class names
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

# Create DataLoaders
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
#transformList.append(transforms.Resize(imgtransCrop))
transformList.append(transforms.RandomResizedCrop(imgtransCrop))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(normalize)      
transformSequence=transforms.Compose(transformList)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

# Paths to the files with training, and validation sets.
# Each file contains pairs (path to image, output vector)
# pathFileTrain = '../CheXpert-v1.0-small/train.csv'
#pathFileTrain = 'CheXpert-v1.0-small/train.csv'
pathFileTrain = 'CheXpert-v1.0-small/multiViewCSV.csv'
pathFileValid = 'CheXpert-v1.0-small/valid.csv'

#LOAD DATASET
#dataset = CheXpertDataSet(pathFileTrain ,transformSequence, policy=policy)
#dataset = CheXpertDataSet(pathFileTrain,transformSequence, policy=policy)
datasetTest = CheXpertDataSet(pathFileTrain ,transformSequence, policy=policy)
#datasetTest, datasetTrain = random_split(dataset, [5000, len(dataset) - 5000])

datasetValid = CheXpertDataSet(pathFileValid, transformSequence)            

#dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
dataLoaderTest = DataLoader(dataset=datasetTest, num_workers=24, pin_memory=True)

# initialize and load the model
modelPA = mod.getmodel(modelName,nnClassCount)
modelPA = torch.nn.DataParallel(modelPA).cuda()

modelAP = mod.getmodel(modelName,nnClassCount)
modelAP = torch.nn.DataParallel(modelAP).cuda()

modelLat = mod.getmodel(modelName,nnClassCount)
modelLat = torch.nn.DataParallel(modelLat).cuda()


class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

print("about to test")
outGT1, outPRED1 = CheXpertTrainer.testMulti(CheXpertTrainer, modelPA, modelAP, modelLat, dataLoaderTest, nnClassCount, paCheckpoint, apCheckpoint, latCheckpoint, class_names)
# outGT3, outPRED3 = CheXpertTrainer.test(CheXpertTrainer, model, dataLoaderTest, nnClassCount, zerosModeltoTest, class_names)
#outGT4, outPRED4 = CheXpertTrainer.test(model, dataLoaderTest, nnClassCount, "model4.pth.tar", class_names)

