#loading libraries

import os
import numpy as np
import sys
import cv2
import time
import matplotlib.pyplot as plt
plt.switch_backend('agg')
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
from train import CheXpertTrainer
import models as mod
from heatmap import HeatmapGenerator
import torchvision
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn as nn
import time
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.metrics.ranking import roc_auc_score
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Paths to the files with training, and validation sets.
# Each file contains pairs (path to image, output vector)
#pathFileTrain = '../CheXpert-v1.0-small/train.csv'
pathFileTrain = 'CheXpert-v1.0-small/train.csv'
pathFileTrainFrontalPa = 'train_frontal_pa.csv'
pathFileTrainFrontalAp = 'train_frontal_ap.csv'
pathFileValid = 'CheXpert-v1.0-small/valid.csv'

# Neural network parameters:
nnIsTrained = False                 #pre-trained using ImageNet
nnClassCount = 14                   #dimension of the output

# Training settings: batch size, maximum number of epochs
# ["DenseNet121","Vgg16","Vgg19"]
modelName = "DenseNet121"
policy = "ones"
trBatchSize = 10
testBatchSize = 5
trMaxEpoch = 3
action = "train" # train or test
#onesModeltoTest = "checkpoints/Vgg19-ones/m-epoch2-Vgg19-ones-27052019-010504.pth.tar"
#onesModeltoTest = "checkpoints/mixedTrainModels/DenseNet/model_ones_3epoch_densenet.tar"
#zerosModeltoTest = "m-epoch2-Vgg19-zeros-260019-135938.pth.tar"

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

#LOAD DATASET
#dataset = CheXpertDataSet(pathFileTrain ,transformSequence, policy=policy)
#dataset = CheXpertDataSet(pathFileTrainFrontalAp,transformSequence, policy=policy)
dataset = CheXpertDataSet(pathFileTrain,transformSequence, policy=policy)



datasetTest, datasetTrain = random_split(dataset, [500, len(dataset) - 500])
#datasetTest, datasetTrain = random_split(dataset, [5000, len(dataset) - 5000])

#datasetTest = torch.load("test.txt")

datasetValid = CheXpertDataSet(pathFileValid, transformSequence)            

dataLoaderTrain = DataLoader(dataset=datasetTest, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
#dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=len(dataset)-500, shuffle=True,  num_workers=24, pin_memory=True)
#dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
dataLoaderTest = DataLoader(dataset=datasetTest, batch_size = testBatchSize, shuffle = True, num_workers=24, pin_memory=True)
#dataLoaderTest = DataLoader(dataset=datasetTest, batch_size = 500, num_workers=24, pin_memory=True)

class DenseNet121(nn.Module):
    """
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        self.densenet121.classifier = nn.Sequential()
        #num_ftrs = self.densenet121.classifier.in_features
        #self.densenet121.classifier = nn.Sequential(
         #   nn.Linear(num_ftrs, out_size),
         #   nn.Sigmoid()
        #)

    def forward(self, x):
        x = self.densenet121(x)
        return x


model = torch.nn.DataParallel(DenseNet121(nnClassCount)).cuda()
model.eval()

train_features = None
train_labels = None

print('works')
for batchID, (varInput, target) in enumerate(dataLoaderTrain):        
    varTarget = target.cuda(non_blocking = True)
    if varInput.shape[0] != trBatchSize:
        continue
    print('here')
    varOutput = model(varInput)
    print('took time')
    if(batchID == 0):
        train_labels = varTarget.detach().cpu().clone()
        train_features = varOutput.detach().cpu().clone()
    else:
        train_labels = torch.cat((train_labels, varTarget.detach().cpu().clone()),0) 
        train_features = torch.cat((train_features, varOutput.detach().cpu().clone()),0)

    
#shape of train_features
#batch by num_features 
#batch by nnClassCount 



test_features = None
test_labels = None

for batchID, (varInput, target) in enumerate(dataLoaderTest):
    if(batchID < 5):       
        varTarget = target.cuda(non_blocking = True)
        if varInput.shape[0] != testBatchSize:
            continue
        varOutput = model(varInput)
        if(batchID == 0):
            test_labels = varTarget.detach().cpu().clone()
            test_features = varOutput.detach().cpu().clone()
        else:
            test_labels = torch.cat((train_labels, varTarget.detach().cpu().clone()),0) 
            test_features = torch.cat((train_features, varOutput.detach().cpu().clone()),0)

print('got features')
test_pred_labels = None

for i in range(nnClassCount):
    unique_labels = np.unique(train_labels[:,i])
    randindex = np.random.randint(low = 0,high= len(train_labels)-1)
    if(len(unique_labels) == 1):
        if(1 in unique_labels):
            train_labels[randindex,i] = 0
        else:
            train_labels[randindex,i] = 1
    clf_gini = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=32, min_samples_leaf=5)
    clf_gini.fit(train_features, train_labels[:,i])
    test_pred = torch.from_numpy(clf_gini.predict(test_features))
    print(type(test_pred), 'test_pred')
    if(i == 0):
        test_pred_labels = test_pred
        test_pred_labels = test_pred_labels.reshape(len(test_pred_labels),1)
    else:
        print((test_pred_labels).shape, 'test_pred_labels')
        test_pred = test_pred.reshape(len(test_pred),1)
        print((test_pred).shape, 'test_pred_labels')

        test_pred_labels = torch.cat((test_pred_labels, test_pred),1)

outAUROC = []   
#test_labels = test_labels.detach().cpu().clone().numpy()
#test_pred_labels = test_pred_labels.detach().cpu().clone().numpy()
for i in range(nnClassCount):
    try:
        outAUROC.append(roc_auc_score(test_labels[:, i], test_pred_labels[:, i]))

    except ValueError:
        pass
aurocMean = np.array(outAUROC).mean()
print(aurocMean, 'aurocMean')
aurocMean1 = np.array(outAUROC)
print(aurocMean1, 'aurocMean')

for i in range(nnClassCount):
    fpr, tpr, threshold = metrics.roc_curve(test_labels.cpu()[:,i], test_pred_labels.cpu()[:,i])
    roc_auc = metrics.auc(fpr, tpr)
    #f = plt.subplot(2, 7, i+1)
    # fpr2, tpr2, threshold2 = metrics.roc_curve(outGT3.cpu()[:,i], outPRED3.cpu()[:,i])
    # roc_auc2 = metrics.auc(fpr2, tpr2)
    #fpr3, tpr3, threshold2 = metrics.roc_curve(outGT4.cpu()[:,i], outPRED4.cpu()[:,i])
    #roc_auc3 = metrics.auc(fpr3, tpr3)


    plt.title('ROC for: '+ modelName + "-" + class_names[i])
    print("ROC for: "+ modelName + "-" + class_names[i] + " ones- %0.2f" % roc_auc)
    # print("ROC for: "+ modelName + "-" + class_names[i] + " zeros- %0.2f" % roc_auc2)
    plt.plot(fpr, tpr, label = 'U-ones: AUC = %0.2f' % roc_auc)
    # plt.plot(fpr2, tpr2, label = 'U-zeros: AUC = %0.2f' % roc_auc2)
    #plt.plot(fpr3, tpr3, label = 'AUC = %0.2f' % roc_auc3)

    plt.legend(loc = 'lower right')
    #plt.plot([0, 1], [0, 1],'r--')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('ROC_'+modelName + "_" + class_names[i]+".png")




