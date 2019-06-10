import torch
import torch.optim as optim
import torch.nn as nn
import time
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.metrics.ranking import roc_auc_score
#from main import dataLoaderTest, dataLoaderTrain, dataLoaderVal, trMaxEpoch, nnClassCount

use_gpu = torch.cuda.is_available()

class CheXpertTrainer():
    
    def testMulti(self, modelPA, modelAP, modelLat, dataLoaderTest, nnClassCount, paCheckpoint, apCheckpoint, latCheckpoint, class_names):   
        
        cudnn.benchmark = True
        
        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            # model.load_state_dict(modelCheckpoint)

            model.load_state_dict(modelCheckpoint['state_dict'], strict=False)


        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
       
        model.eval()
        
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):

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