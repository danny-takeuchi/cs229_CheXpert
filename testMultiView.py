import torch
import torch.optim as optim
import torch.nn as nn
import time
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.metrics.ranking import roc_auc_score
from collections import defaultdict
#from main import dataLoaderTest, dataLoaderTrain, dataLoaderVal, trMaxEpoch, nnClassCount

use_gpu = torch.cuda.is_available()

class CheXpertTrainer():
    
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

        patientsDict = defaultdict(list) 
        with torch.no_grad():
            for i, (input, target, patient, study, view) in enumerate(dataLoaderTest):
                dictKey = (patient[0], study[0])
                patientsDict[dictKey].append((input, view, target))
            for patientStudy, infoList in patientsDict.items():
                for patientInfo in infoList:
                    print(infoList[2])
                    target = target.cuda()
                    outGT = torch.cat((outGT, target), 0).cuda()

                    bs, c, h, w = input.size()
                    varInput = input.view(-1, c, h, w)
                
                    out = model(varInput)
                    outPRED = torch.cat((outPRED, out), 0)
        aurocIndividual = CheXpertTrainer.computeAUROC(self, outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (class_names[i], ' ', aurocIndividual[i])
        
        return outGT, outPRED