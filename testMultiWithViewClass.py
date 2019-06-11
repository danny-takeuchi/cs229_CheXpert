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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class DenseNet121(nn.Module):
    """
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
    
class Vgg16(nn.Module):
    def __init__(self, out_size):
        super(Vgg16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        num_ftrs = self.vgg16.classifier[0].in_features
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg16(x)
        return x


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

                image_names.append("" + image_name)
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

pathFileTrain = 'train-small.csv'
# pathFileTrain = 'CheXpert-v1.0-small/train.csv'
pathFileValid = 'CheXpert-v1.0-small/valid.csv'

trBatchSize = 4

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = (320, 320)
imgtransCrop = 224

# Create DataLoaders
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.Resize(imgtransCrop))
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

trainloader = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
testloader = DataLoader(dataset=datasetTest, num_workers=24, pin_memory=True)

net = torch.nn.DataParallel(Vgg16(3)).cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if inputs.shape[0] != trBatchSize:
               continue
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        labels = labels.cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(i, loss.item())
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        labels = labels.cuda()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

class CheXpertTrainer():

    def computeAUROC (self, dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        return outAUROC
    
    def testMulti(self, modelPA, modelAP, modelLat, dataLoaderTest, nnClassCount, paCheckpoint, apCheckpoint, latCheckpoint, class_names):   
        
        cudnn.benchmark = True
        print("In testMulti")
        
        if paCheckpoint != None and use_gpu:
            modelPACheckpoint = torch.load(paCheckpoint)
            modelAPCheckpoint = torch.load(apCheckpoint)
            modelLatCheckpoint = torch.load(latCheckpoint)
            # model.load_state_dict(modelCheckpoint)

            modelPA.load_state_dict(modelPACheckpoint['state_dict'], strict=False)
            modelAP.load_state_dict(modelAPCheckpoint['state_dict'], strict=False)
            modelLat.load_state_dict(modelLatCheckpoint['state_dict'], strict=False)



        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
       
        modelPA.eval()
        modelAP.eval()
        modelLat.eval()
        print("about to loop")
        patientsDict = defaultdict(list) 
        with torch.no_grad():
            for i, (input, target, patient, study, view) in enumerate(dataLoaderTest):
                dictKey = (patient[0], study[0])
                patientsDict[dictKey].append((input, view, target))
                if (i % 1000) == 0:
                    print(i)
            print("finished looping!")
            patientCounter = 0
            for patientStudy, infoList in patientsDict.items():
                patientCounter += 1
                predictionsForPatientStudy = []
                for patientInfo in infoList:
                    input = patientInfo[0]
                    #viewType = patientInfo[1][0]
                    viewType = net(input)
                    target = patientInfo[2]
                    target = target.cuda()
                    outGT = torch.cat((outGT, target), 0).cuda()
                    bs, c, h, w = input.size()
                    varInput = input.view(-1, c, h, w)
                    if viewType == 0:
                        out = modelAP(varInput)
                    elif viewType == 1:
                        out = modelPA(varInput)
                    else:
                        out = modelLat(varInput)
                    predictionsForPatientStudy.append(out)
                for prediction in predictionsForPatientStudy:
                    newPrediction = torch.mean(torch.stack(predictionsForPatientStudy), dim=0)
                    outPRED = torch.cat((outPRED, newPrediction), 0) 
                if (patientCounter % 100) == 0:
                    print(patientCounter)
        aurocIndividual = CheXpertTrainer.computeAUROC(self, outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (class_names[i], ' ', aurocIndividual[i])
        
        return outGT, outPRED

