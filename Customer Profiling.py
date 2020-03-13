# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np 
import pandas as pd 
from matplotlib import style
style.use('ggplot')
from sklearn import preprocessing
colors = 10 * ["g", "r", "c", "b", "k"]
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_excel(r"C:\Users\Christopher Masloub\Downloads\default of credit card clients (1).xls")
df = df.drop("ID", axis = 1 )
df.head()
x = df
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
data_Normalized = min_max_scaler.fit_transform(x)


class Improved_K_Means:

    def __init__(self, k=5, tol=0.001, max_iter=300):

        self.k = k

        self.tol = tol

        self.max_iter = max_iter



    def fit(self, data):


        self.centroids = {}



        for i in range(self.k):

            self.centroids[i] = data[i]



        for i in range(self.max_iter):

            self.classifications = {}



            for i in range(self.k):

                self.classifications[i] = []



            for featureset in data:

                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]

                classification = distances.index(min(distances))

                self.classifications[classification].append(featureset)



            prev_centroids = dict(self.centroids)



            for classification in self.classifications:

                self.centroids[classification] = np.average(self.classifications[classification], axis=0)



            optimized = True



            for c in self.centroids:

                original_centroid = prev_centroids[c]

                current_centroid = self.centroids[c]

                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:

                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))

                    optimized = False



            if optimized:

                break



    def predict(self, data):

        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]

        classification = distances.index(min(distances))

        return classification



    def update(self, new_data, delta):

        for featureset in new_data:

            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]



            if min(distances) < delta:

                classification = distances.index(min(distances))

                self.classifications[classification].append(featureset)

                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            else:

                self.centroids[self.k] = featureset

                self.classifications[self.k] = []

                self.classifications[self.k].append(featureset)

                self.k = self.k + 1
                
                

np.seterr(divide='ignore', invalid='ignore')
run_Clusters = Improved_K_Means()
run_Clusters.fit(data_Normalized) 




i = 1    
       
for classification in run_Clusters.classifications:
    print(classification)      
    with open('ClassificationData1.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if i == 1: 
            temp = list(df.columns)
            temp.append("Cluster")
            writer.writerow(temp)
            i = 0
        
        for featureset in run_Clusters.classifications[classification]:
           List = []
           for feature in featureset: 
               List.append(feature)
           List.append(classification)    
              
           writer.writerow(List)
        

df2 = pd.read_csv('ClassificationData.csv')


target = df2.Cluster
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
labelencoder_Y_1 = LabelEncoder()
target = labelencoder_Y_1.fit_transform(target)
target = to_categorical(target)
features = df2.drop('Cluster', axis = 1)

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.75)




import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


model = Sequential()
model.add(Dense(30, activation='relu', input_dim=24))
model.add(Dropout(0.5))
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=100,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)



    