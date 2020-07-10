import os
import argparse
import itertools
import pandas as pd
import numpy as np
import torch
import random

from pathlib import Path
from tqdm import tqdm
from shutil import copyfile
from collections import defaultdict
from sklearn.metrics.ranking import roc_auc_score
from sklearn.tree import DecisionTreeClassifier



DATA_DIR = Path("/deep/group")
CHEXPERT_PARENT_DATA_DIR = DATA_DIR / "CheXpert"
COL_REL_PATH = 'Relative_Path'
COL_PATH = 'Path'
COL_STUDY = "Study"
COL_PATIENT = "Patient"
CHEXPERT_TASKS = ["No Finding",
                  "Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Pneumonia",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Pleural Other",
                  "Fracture",
                  "Support Devices"]

TASKS_TO_REMOVE = ["Enlarged Cardiomediastinum",
                   "Airspace Opacity",
                   "Pneumonia", 
                   "Support Devices"]

input_path = '/deep/group/RareXpert/valid_absolute.csv'
output_dir = '/deep/u/ktran/spr-rare/CheXpert/'

def load_df(input_path):
    df = pd.read_csv(Path(input_path))
    df[COL_REL_PATH] = df[COL_PATH]
    # Prepend the data dir to get the full path.
    df[COL_PATH] = df[COL_PATH].apply(lambda x: CHEXPERT_PARENT_DATA_DIR / x)
    df[COL_STUDY] = df[COL_PATH].apply(lambda p: Path(p).parent)
    df[COL_PATIENT] = df[COL_STUDY].apply(lambda p: Path(p).parent)
    df = df.rename(columns={"Lung Opacity": "Airspace Opacity"}).sort_values(COL_STUDY)
    df[CHEXPERT_TASKS] = df[CHEXPERT_TASKS].fillna(value=0)
    return df

def set_study_as_index(df):
    df.index = df[COL_STUDY]

def set_patient_as_index(df):
    df.index = df[COL_PATIENT]

def reset_index(df):
    df_copy = df.copy()
    df_copy = df.reset_index(drop=True)
    return df_copy 

# def get_labels(df):
#     # study_df = df.drop_duplicates(subset=COL_STUDY)
#     labels = df[CHEXPERT_TASKS]
#     return labels

def remove_tasks(labels):
    for task in TASKS_TO_REMOVE:
        labels.drop(task, axis=1, inplace=True)

def get_paths(df):
    return df[COL_PATH]

def get_patients(df):
    return df[COL_PATIENT].drop_duplicates()

def get_studies(df):
    return df[COL_STUDY].drop_duplicates()

df = load_df(input_path)
# patients = get_patients(df)
# set_patient_as_index(df)
# studies = get_studies(df)
# set_study_as_index(df) 
labels = df.copy()[CHEXPERT_TASKS]
# img_paths = get_paths(df)
remove_tasks(labels)
# reset_index(labels)

seen_set = set([0, 1, 4, 7])

df = df.assign(Unseen = [0]*len(df))

labels_list = []
for i in range(len(df)):
  # print(labels)
  # print(i)
  unseen = False
  label = torch.FloatTensor(labels.iloc[i])
  for j in range(10):
    if label[j] == 1:
      if j not in seen_set:
        df.at[i, 'Unseen'] = 1
        unseen = True
  if unseen:
    labels_list.append(1)
  else:
    labels_list.append(0)

labels = np.array(labels_list) 
labels = torch.from_numpy(labels)


print(labels.shape)

# df = df.drop(COL_REL_PATH,axis=1)
# df = df.drop(COL_STUDY, axis=1)
# df = df.drop(COL_PATIENT, axis=1)

# df.to_csv('/deep/group/RareXpert/valid_logistic.csv',index=True,header=True)

probs_df = pd.read_csv(Path('/deep/group/RareXpert/cam_val.csv'))
probs = torch.tensor(probs_df.values)
probs = probs[:,1:]

print(probs.shape)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=32, min_samples_leaf=5)
clf_gini.fit(probs[:200,], labels[:200])
test_pred = torch.from_numpy(clf_gini.predict(probs[200:,:]))
score = roc_auc_score(labels[200:], test_pred)
print(score)


seen_set = set([0, 1, 4, 7])

probs_df = probs_df.assign(Unseen = [0]*len(probs_df))

for i in range(len(probs_df)):
  label = torch.FloatTensor(labels_df.iloc[i])
  for j in range(10):
    if label[j] == 1:
      if j not in seen_set:
        probs_df.at[i, 'Unseen'] = 1


cams_df = pd.read_csv('/deep/group/RareXpert/cam_val.csv')
val_df = pd.read_csv('/deep/group/RareXpert/valid_absolute.csv')






















