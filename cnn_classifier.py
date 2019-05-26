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

from dataset import CheXpertDataSet
from train_cnn import CheXpertTrainer
import models as mod
from heatmap import HeatmapGenerator

class CheXpertCnnClassifier(Dataset):
    def __init__(self, image_list_file, transform=None):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        image_names = []
        labels = []

        with open(image_list_file, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None)
            k = 0
            for line in csvReader:
                k += 1
                image_name = line[0]
                if line[3] == "Lateral":
                    label = 1
                else:
                    if line[4] == "AP":
                        label = 2
                    else:
                        label = 0

                image_names.append("../" + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""

        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_names)

    # Paths to the files with training, and validation sets.
# Each file contains pairs (path to image, output vector)
pathFileTrain = 'train-small.csv'

# pathFileTrain = 'CheXpert-v1.0-small/train.csv'
pathFileValid = '../CheXpert-v1.0-small/valid.csv'

# Neural network parameters:
nnIsTrained = False                 #pre-trained using ImageNet
nnClassCount = 3                  #dimension of the output

# Training settings: batch size, maximum number of epochs
# ["DenseNet121","Vgg16","Vgg19"]
modelName = "DenseNet121"
policy = "ones"
trBatchSize = 64
trMaxEpoch = 3
action = "train" # train or test
testModel = "checkpoints/cnn/trial1"

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = (320, 320)
imgtransCrop = 224

# Class names
class_names = ['AP', 'L', 'PA']

# Create DataLoaders
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
#transformList.append(transforms.Resize(imgtransCrop))
transformList.append(transforms.RandomResizedCrop(imgtransCrop))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(normalize)
transformSequence=transforms.Compose(transformList)

#LOAD DATASET
# dataset = CheXpertViewDataSet("CheXpert-v1.0-small/train.csv", transformSequence)
dataset = CheXpertCnnClassifier("train-small.csv", transformSequence)
datasetTest, datasetTrain = random_split(dataset, [500, len(dataset) - 500])
datasetValid = CheXpertCnnClassifier(pathFileValid, transformSequence)

dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
dataLoaderTest = DataLoader(dataset=datasetTest, num_workers=24, pin_memory=True)

model = mod.getmodel(modelName,nnClassCount)
model = torch.nn.DataParallel(model).cuda()

if action == "train":
    # train the model
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    batch, losst, losse = CheXpertTrainer.train(CheXpertTrainer, model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, timestampLaunch, checkpoint = None, modelName=modelName,policy=policy)
    print("Model trained")

    #print loss metrics
    losstn = []
    for i in range(0, len(losst), 35):
        losstn.append(np.mean(losst[i:i+35]))
    print(losstn)
    print(losse)


outGT1, outPRED1 = CNNTrainer.test(CNNTrainer, model, dataLoaderTest, nnClassCount, testModel, class_names)
#outGT3, outPRED3 = CNNTrainer.test(CNNTrainer, model, dataLoaderTest, nnClassCount, testModel, class_names)
#outGT4, outPRED4 = CheXpertTrainer.test(model, dataLoaderTest, nnClassCount, "model4.pth.tar", class_names)

for i in range(nnClassCount):
    fpr, tpr, threshold = metrics.roc_curve(outGT1.cpu()[:,i], outPRED1.cpu()[:,i])
    roc_auc = metrics.auc(fpr, tpr)
    #f = plt.subplot(2, 7, i+1)
    fpr2, tpr2, threshold2 = metrics.roc_curve(outGT3.cpu()[:,i], outPRED3.cpu()[:,i])
    roc_auc2 = metrics.auc(fpr2, tpr2)
    #fpr3, tpr3, threshold2 = metrics.roc_curve(outGT4.cpu()[:,i], outPRED4.cpu()[:,i])
    #roc_auc3 = metrics.auc(fpr3, tpr3)


    plt.title('ROC for: '+ modelName + "-" + class_names[i])
    print("ROC for: "+ modelName + "-" + class_names[i] + " ones- %0.2f" % roc_auc)
    print("ROC for: "+ modelName + "-" + class_names[i] + " zeros- %0.2f" % roc_auc2)
    plt.plot(fpr, tpr, label = 'U-ones: AUC = %0.2f' % roc_auc)
    plt.plot(fpr2, tpr2, label = 'U-zeros: AUC = %0.2f' % roc_auc2)
    #plt.plot(fpr3, tpr3, label = 'AUC = %0.2f' % roc_auc3)

    plt.legend(loc = 'lower right')
    #plt.plot([0, 1], [0, 1],'r--')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('ROC_'+modelName + "_" + class_names[i]+".png")









