#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Name: Muhammad Athar
# Roll #: 18I-0692

import numpy as np
import pandas as pd
import csv
import math
import copy
                                       # K-Nearest Neighbours (KNN) Classification

# Dataset: Car Evaluation
# Classes: 4 (unacc, acc, good, vgood)
# Features / Attributes: 6
# Records: 1728

def keyfunc(s):
    return s[1]

def find_distance_with_all_neighbours(val_or_test_dataset_record, train_dataset_record):
    sum_of_squares = 0
    euclidean_dist = 0
    
    for k in range(0, 6):
        diff = train_dataset_record[k] - val_or_test_dataset_record[k]
        
        square = diff ** 2
        
        sum_of_squares = sum_of_squares + square
        
    euclidean_dist = math.sqrt(sum_of_squares)
    
    return euclidean_dist
        
def find_accuracy_with_some_k(k, original_dataset_numerical, train_dataset_numerical, validation_dataset_numerical):
    euc_distance_with_all_neighbours = [0 for i in range(len(train_dataset_numerical))]
    
    least_distant_neigh_indices = [0 for i in range(k)]
    
    car_classes = [0, 0, 0, 0]
    
    accuracy = 0
    
    matched = 0
    not_matched = 0
    
    # Predicting Classes of each record of Validation Dataset
    for i in range(0, len(validation_dataset_numerical)):
        for j in range(0, len(train_dataset_numerical)):
            euc_distance_with_all_neighbours[j] = find_distance_with_all_neighbours(validation_dataset_numerical[i], train_dataset_numerical[j])
        
        enu_euc_dist = enumerate(euc_distance_with_all_neighbours)
        
        enu_euc_dist = list(enu_euc_dist)
        
        result = sorted(enu_euc_dist, key=keyfunc)
        
        for m in range(0, k):
            least_distant_neigh_indices[m] = result[m][0]
        
        car_classes = [0, 0, 0, 0]
        
        for n in range(k):
            if (train_dataset_numerical[least_distant_neigh_indices[n]][6] == 1):
                car_classes[0] = car_classes[0] + 1   
            elif train_dataset_numerical[least_distant_neigh_indices[n]][6] == 2:
                car_classes[1] = car_classes[1] + 1
            elif train_dataset_numerical[least_distant_neigh_indices[n]][6] == 3:
                car_classes[2] = car_classes[2] + 1
            elif train_dataset_numerical[least_distant_neigh_indices[n]][6] == 4:
                car_classes[3] = car_classes[3] + 1
        
        car_classes_enumerate = enumerate(car_classes)
        
        car_classes_enumerate = list(car_classes_enumerate)
            
        result1 = sorted(car_classes_enumerate, key=keyfunc)
        
        which_class = result1[3][0]
        
        validation_dataset_numerical[i][6] = which_class + 1
    
    xy = 1000
    
    for i in range(0, len(validation_dataset_numerical)):
        if original_dataset_numerical[xy][6] == validation_dataset_numerical[i][6]:
            matched = matched + 1
            xy+=1
        else:
            not_matched = not_matched + 1
            xy+=1
    
    accuracy = (matched / len(validation_dataset_numerical)) * 100
                                         
    return accuracy
    
def change_categorical_values_to_numerical(dataset):
    dataset_len = len(dataset)
    
    for i in range(0, dataset_len):
        for j in range(0, 7):
            # First Column - buying
            if j == 0:
                if dataset[i][j] == 'vhigh':
                    dataset[i][j] = 4
                elif dataset[i][j] == 'high':
                    dataset[i][j] = 3
                elif dataset[i][j] == 'med':
                    dataset[i][j] = 2
                elif dataset[i][j] == 'low':
                    dataset[i][j] = 1
            
            # 2nd Column - maint
            elif j == 1:
                if dataset[i][j] == 'vhigh':
                    dataset[i][j] = 4
                elif dataset[i][j] == 'high':
                    dataset[i][j] = 3
                elif dataset[i][j] == 'med':
                    dataset[i][j] = 2
                elif dataset[i][j] == 'low':
                    dataset[i][j] = 1
                    
            # 3rd Column - doors
            elif j == 2:
                if dataset[i][j] == '2':
                    dataset[i][j] = 2
                if dataset[i][j] == '3':
                    dataset[i][j] = 3
                if dataset[i][j] == '4':
                    dataset[i][j] = 4
                if dataset[i][j] == '5more':
                    dataset[i][j] = 5
                    
            # 4th Column - persons
            elif j == 3:
                if dataset[i][j] == '2':
                    dataset[i][j] = 2
                if dataset[i][j] == '3':
                    dataset[i][j] = 3
                if dataset[i][j] == '4':
                    dataset[i][j] = 4
                if dataset[i][j] == 'more':
                    dataset[i][j] = 5
                    
            # 5th Column - lug_boot
            elif j == 4:
                if dataset[i][j] == 'small':
                    dataset[i][j] = 1
                elif dataset[i][j] == 'med':
                    dataset[i][j] = 2
                elif dataset[i][j] == 'big':
                    dataset[i][j] = 3
                    
            # 6th Column - safety
            elif j == 5:
                if dataset[i][j] == 'low':
                    dataset[i][j] = 1
                elif dataset[i][j] == 'med':
                    dataset[i][j] = 2
                elif dataset[i][j] == 'high':
                    dataset[i][j] = 3
                    
            # 7th Column - CLASS
            elif j == 6:
                if dataset[i][j] == 'unacc':
                    dataset[i][j] = 1
                elif dataset[i][j] == 'acc':
                    dataset[i][j] = 2
                elif dataset[i][j] == 'good':
                    dataset[i][j] = 3
                elif dataset[i][j] == 'vgood':
                    dataset[i][j] = 4

def find_accuracy_of_test_dataet(confusion_matrix, k, original_dataset_numerical, train_dataset_numerical, test_dataset_numerical):
    euc_distance_with_all_neighbours = [0 for i in range(len(train_dataset_numerical))]
    
    least_distant_neigh_indices = [0 for i in range(k)]
    
    car_classes = [0, 0, 0, 0]
    
    accuracy = 0
    
    matched = 0
    not_matched = 0
    
    # Predicting Classes of each record of Test Dataset
    for i in range(0, len(test_dataset_numerical)):
        for j in range(0, len(train_dataset_numerical)):
            euc_distance_with_all_neighbours[j] = find_distance_with_all_neighbours(test_dataset_numerical[i], train_dataset_numerical[j])
        
        enu_euc_dist = enumerate(euc_distance_with_all_neighbours)
        
        enu_euc_dist = list(enu_euc_dist)
        
        result = sorted(enu_euc_dist, key=keyfunc)
        
        for m in range(0, k):
            least_distant_neigh_indices[m] = result[m][0]
        
        car_classes = [0, 0, 0, 0]
        
        for n in range(k):
            if (train_dataset_numerical[least_distant_neigh_indices[n]][6] == 1):
                car_classes[0] = car_classes[0] + 1   
            elif train_dataset_numerical[least_distant_neigh_indices[n]][6] == 2:
                car_classes[1] = car_classes[1] + 1
            elif train_dataset_numerical[least_distant_neigh_indices[n]][6] == 3:
                car_classes[2] = car_classes[2] + 1
            elif train_dataset_numerical[least_distant_neigh_indices[n]][6] == 4:
                car_classes[3] = car_classes[3] + 1
        
        car_classes_enumerate = enumerate(car_classes)
        
        car_classes_enumerate = list(car_classes_enumerate)
            
        result1 = sorted(car_classes_enumerate, key=keyfunc)
        
        which_class = result1[3][0]
        
        test_dataset_numerical[i][6] = which_class + 1
    
    xy = 1000
    
    for i in range(0, len(test_dataset_numerical)):
        org_class = original_dataset_numerical[xy][6] - 1
        predicted_class = test_dataset_numerical[i][6] - 1
        
        confusion_matrix[predicted_class][org_class]+=1
        
        if original_dataset_numerical[xy][6] == test_dataset_numerical[i][6]:
            matched = matched + 1
            xy+=1
        else:
            not_matched = not_matched + 1
            xy+=1
    
    accuracy = (matched / len(test_dataset_numerical)) * 100
                                         
    return accuracy
    
def load_and_split_dataset(dataset_fname, original_dataset=[], train=[], validation=[], test=[]):
    
    # We have 1728 records. We will give 1000 records to Train Dataset, 300 records to Validation dataset and 428 records
    # to Test Dataset.
    
    # Reading car.data dataset file in text mode.
    with open(dataset_fname, 'r') as csvfile:
        lines = csv.reader(csvfile)
        
        org_dataset = list(lines)
        
        for i in range(0, len(org_dataset)):
            if i < 1000:
                train.append(copy.deepcopy(org_dataset[i]))
                original_dataset.append(copy.deepcopy(org_dataset[i]))
            if i >= 1000 and i < 1300:
                validation.append(copy.deepcopy(org_dataset[i]))
                original_dataset.append(copy.deepcopy(org_dataset[i]))
            if i >= 1300 and i < 1728:
                test.append(copy.deepcopy(org_dataset[i]))
                original_dataset.append(copy.deepcopy(org_dataset[i]))
                
original_dataset = []

# Original Dataset divided to 3 parts i.e. Train Dataset, Validation Dataset, Test Dataset
train_dataset = []
validation_dataset = []
test_dataset = []

load_and_split_dataset('Assignment3/car.data', original_dataset, train_dataset, validation_dataset, test_dataset)

original_dataset_numerical = []
train_dataset_numerical = []
validation_dataset_numerical = []
test_dataset_numerical = []

for i in range(0, len(original_dataset)):
        original_dataset_numerical.append(copy.deepcopy(original_dataset[i]))
        
for i in range(0, len(train_dataset)):
        train_dataset_numerical.append(copy.deepcopy(train_dataset[i]))

for i in range(0, len(validation_dataset)):
        validation_dataset_numerical.append(copy.deepcopy(validation_dataset[i]))

for i in range(0, len(test_dataset)):
        test_dataset_numerical.append(copy.deepcopy(test_dataset[i]))

# Changing Categorical Data to Numerical Data
change_categorical_values_to_numerical(original_dataset_numerical)
change_categorical_values_to_numerical(train_dataset_numerical)
change_categorical_values_to_numerical(validation_dataset_numerical)
change_categorical_values_to_numerical(test_dataset_numerical)

for i in range(0, len(validation_dataset)):
    validation_dataset_numerical[i][6] = -1

for i in range(0, len(test_dataset)):
    test_dataset_numerical[i][6] = -1

flag = True

previous_accuracy = 0
kk = -1
val_dataset_accuracy_ = -10
confusion_matrix = [[0 for i in range(4)] for j in range(4)]

while flag == True:
    kk+=2
    
    val_dataset_accuracy_ = find_accuracy_with_some_k(kk, original_dataset_numerical, train_dataset_numerical, validation_dataset_numerical)
    
    if val_dataset_accuracy_ > previous_accuracy:
        flag = True
        previous_accuracy = val_dataset_accuracy_
    else:
        flag = False

test_dataset_accuracy = find_accuracy_of_test_dataet(confusion_matrix, kk, original_dataset_numerical, train_dataset_numerical, test_dataset_numerical)

print("Tuned K Value: ", kk)
print("\nValidation Dataset Accuray: ", val_dataset_accuracy_)
print("Test Dataset Accuracy: ", test_dataset_accuracy)

# For Label 1
l1_TP = confusion_matrix[0][0]
l1_TN = confusion_matrix[1][1] + confusion_matrix[2][1] + confusion_matrix[3][1] + confusion_matrix[1][2] + confusion_matrix[2][2] + confusion_matrix[3][2] + confusion_matrix[1][3] + confusion_matrix[2][3] + confusion_matrix[3][3]
l1_FP = confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][3]
l1_FN = confusion_matrix[1][0] + confusion_matrix[2][0] + confusion_matrix[3][0]

l1_accuracy = (l1_TP + l1_TN) / (l1_TP + l1_TN + l1_FP + l1_FN)
l1_precision = l1_TP / (l1_TP + l1_FP)
l1_recall = l1_TP / (l1_TP + l1_FN)
l1_f1_score = 2 * ((l1_precision * l1_recall) / (l1_precision + l1_recall))

print("\nL1 Accuracy: ", l1_accuracy)
print("L1 Precision: ", l1_precision)
print("L1 Recall: ", l1_recall)
print("L1 F1-Score: ", l1_f1_score)

# For Label 2
l2_TP = confusion_matrix[1][1]
l2_TN = confusion_matrix[0][0] + confusion_matrix[0][2] + confusion_matrix[0][3] + confusion_matrix[2][0] + confusion_matrix[2][2] + confusion_matrix[2][3] + confusion_matrix[3][0] + confusion_matrix[3][2] + confusion_matrix[3][3]
l2_FP = confusion_matrix[1][0] + confusion_matrix[1][2] + confusion_matrix[1][3]
l2_FN = confusion_matrix[0][1] + confusion_matrix[2][1] + confusion_matrix[3][1]

l2_accuracy = (l2_TP + l2_TN) / (l2_TP + l2_TN + l2_FP + l2_FN)
l2_precision = l2_TP / (l2_TP + l2_FP)
l2_recall = l2_TP / (l2_TP + l2_FN)
l2_f1_score = 2 * ((l2_precision * l2_recall) / (l2_precision + l2_recall))

print("\nL2 Accuracy: ", l2_accuracy)
print("L2 Precision: ", l2_precision)
print("L2 Recall: ", l2_recall)
print("L2 F1-Score: ", l2_f1_score)

# For Label 3
l3_TP = confusion_matrix[2][2]
l3_TN = confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][3] + confusion_matrix[1][0] + confusion_matrix[1][1] + confusion_matrix[1][3] + confusion_matrix[3][0] + confusion_matrix[3][1] + confusion_matrix[3][3]
l3_FP = confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][3]
l3_FN = confusion_matrix[0][2] + confusion_matrix[1][2] + confusion_matrix[3][2]

l3_accuracy = (l3_TP + l3_TN) / (l3_TP + l3_TN + l3_FP + l3_FN)
#l3_precision = l3_TP / (l3_TP + l3_FP)
l3_recall = l3_TP / (l3_TP + l3_FN)
#l3_f1_score = 2 * ((l3_precision * l3_recall) / (l3_precision + l3_recall))

print("\nL3 Accuracy: ", l3_accuracy)
print("L3 Recall: ", l3_recall)
#print("L3 Precision: ", l3_precision)
#print("L3 F1_Score: ", l3_f1_score)

# For label 4
l4_TP = confusion_matrix[3][3]
l4_TN = confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[1][0] + confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][2]
l4_FP = confusion_matrix[3][0] + confusion_matrix[3][1] + confusion_matrix[3][2]
l4_FN = confusion_matrix[0][3] + confusion_matrix[1][3] + confusion_matrix[2][3]

l4_accuracy = (l4_TP + l4_TN) / (l4_TP + l4_TN + l4_FP + l4_FN)
#l4_precision = l4_TP / (l4_TP + l4_FP)
l4_recall = l4_TP / (l4_TP + l4_FN)
#l4_f1_score = 2 * ((l4_precision * l4_recall) / (l4_precision + l4_recall))

print("\nL4 Accuracy: ", l4_accuracy)
print("L4 Recall: ", l4_recall)
#print("L4 Precision: ", l4_precision)
#print("L4 F1_Score: ", l4_f1_score)

# Mirco - Macro

total_tp = l1_TP + l2_TP + l3_TP + l4_TP
total_tn = l1_TN + l2_TN + l3_TN + l4_TN
total_fp = l1_FP + l2_FP + l3_FP + l4_FP
total_fn = l1_FN + l2_FN + l3_FN + l4_FN

micro_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
micro_precision = total_tp / (total_tp + total_fp)
micro_recall = total_tp / (total_tp + total_fn)
micro_f1_score = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall))

print("\nMirco Accuracy: ", micro_accuracy)
print("Mirco Precision: ", micro_precision)
print("Mirco Recall: ", micro_recall)
print("Mirco F1-Score: ", micro_f1_score)

#macro_f1_score = (l1_f1_score + l2_f1_score + l3_f1_score + l4_f1_score) / 4


# In[ ]:




