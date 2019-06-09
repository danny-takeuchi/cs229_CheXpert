import csv
from PIL import Image
import torch
from torch.utils.data import Dataset

class CheXpertDataSet(Dataset):
    def __init__(self, image_list_file, transform=None, policy="ones"):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        image_names = []
        labels = []
        patients = []
        studies = []

        with open(image_list_file, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None)
            k=0
            for line in csvReader:
                k+=1
                image_name= line[0]
                label = line[5:]
                patient = image_name.split('/', 3)[2]
                study = image_name.split('/', 4)[3]

                
                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                        
                image_names.append(image_name)
                labels.append(label)
                patients.append(patient)
                studies.append(study)

        self.image_names = image_names
        self.labels = labels
        self.patients = patients
        self.studies = studies
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        patient = self.patients[index]
        study = self.studies[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), patient, study 

    def __len__(self):
        return len(self.image_names)
