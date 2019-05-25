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
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
from train import CheXpertTrainer
import models as mod
from heatmap import HeatmapGenerator
class CheXpertViewDataSet(Dataset):
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

                image_names.append(image_name)
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

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(150528, 3)
        self.softmax = torch.nn.Softmax()
    def forward(self, x):
        y_probs = self.linear(x)
        return y_probs


# Paths to the files with training, and validation sets.
# Each file contains pairs (path to image, output vector)
pathFileTrain = 'train-small.csv'
# pathFileTrain = 'CheXpert-v1.0-small/train.csv'
pathFileValid = 'CheXpert-v1.0-small/valid.csv'

# Neural network parameters:
nnIsTrained = False                 #pre-trained using ImageNet
nnClassCount = 14                   #dimension of the output

# Training settings: batch size, maximum number of epochs
# ["DenseNet121","Vgg16","Vgg19"]
modelName = "Vgg19"
policy = "ones"
trBatchSize = 64
trMaxEpoch = 3
action = "test" # train or test
onesModeltoTest = "m-epoch1-Vgg19-ones-26042019-025551.pth.tar"
zerosModeltoTest = "m-epoch2-Vgg19-zeros-26042019-135938.pth.tar"

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = (320, 320)
imgtransCrop = 224

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
dataset = CheXpertViewDataSet("train-small.csv", transformSequence)
datasetTest, datasetTrain = random_split(dataset, [500, len(dataset) - 500])
datasetValid = CheXpertViewDataSet(pathFileValid, transformSequence)

dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
dataLoaderTest = DataLoader(dataset=datasetTest, num_workers=24, pin_memory=True)

model = LinearRegression()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
num_iter = 1
for epoch in range(num_iter):
    for batchID, (varInput, target) in enumerate(dataLoaderTrain):
        #View Type: PA: 0. Lateral:1, AP: 2
        # Forward pass: Compute predicted y by passing
        # x to the model
        varInput = varInput.reshape(64, -1)
        pred_y = model(varInput)

        # Compute and print loss
        loss = criterion(pred_y, target)

        # Zero gradients, perform a backward pass,
        # and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(batchID)
    print('epoch {}, loss {}'.format(epoch, loss.item()))



