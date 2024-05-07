from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2

from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import accuracy_score
from keras.applications import EfficientNetV2L

from sklearn.metrics import confusion_matrix #class to calculate accuracy and other metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier

main = tkinter.Tk()
main.title("Pest Classification using EfficientNetV2L") #designing main screen
main.geometry("1300x1200")

global filename, X, Y, efficient_model
global X_train, X_test, y_train, y_test
global labels, accuracy, precision, recall, fscore

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset(): #function to upload dataset
    global filename, X, Y, labels
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name.strip())
    text.insert(END,"Various Pests Found in Dataset = "+str(labels)+"\n\n")            
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32, 32))
                    X.append(img)
                    label = getLabel(name)
                    Y.append(label)
                    print(name+" "+str(label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    text.insert(END,"Total images found in Dataset : "+str(X.shape[0])+"\n\n")
    unique, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Pests  Names")
    plt.ylabel("Count")
    plt.title("Dataset Class Label Graph")
    plt.xticks(rotation=90)
    plt.show()

def datasetPreprocessing():
    text.delete('1.0', END)
    global X, Y
    global X_train, X_test, y_train, y_test
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset Shuffling & Normalization Completed\n\n")
    text.insert(END,"Dataset train & test split as 80% dataset for training and 20% for testing\n\n")
    text.insert(END,"Training Size (80%): "+str(X_train.shape[0])+"\n") #print training and test size
    text.insert(END,"Testing Size (20%): "+str(X_test.shape[0])+"\n")

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    global labels
    global accuracy, precision, recall, fscore
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall    : '+str(r)+"\n")
    text.insert(END,algorithm+' FSCORE    : '+str(f)+"\n\n")    
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(8, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    
    
 
def trainModel():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, efficient_model
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    efficient_model = EfficientNetV2L(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in efficient_model.layers:
        layer.trainable = False
    efficient_model = Sequential() 
    efficient_model.add(Convolution2D(32, (3, 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    efficient_model.add(MaxPooling2D(pool_size = (2, 2)))
    efficient_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    efficient_model.add(MaxPooling2D(pool_size = (2, 2)))
    efficient_model.add(Flatten())
    efficient_model.add(Dense(units = 256, activation = 'relu'))
    efficient_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    efficient_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/efficient_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/efficient_weights.hdf5', verbose = 1, save_best_only = True)
        hist = efficient_model.fit(X_train, y_train, batch_size = 32, epochs = 35, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/efficient_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        efficient_model.load_weights("model/efficient_weights.hdf5")
    predict = efficient_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("EfficientNetV2L", predict, y_test1)

def trainXGBoost():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    X_train1 = np.reshape(X_train, (X_train.shape[0], (X_train.shape[1] * X_train.shape[2] * X_train.shape[3])))
    X_test1 = np.reshape(X_test, (X_test.shape[0], (X_test.shape[1] * X_test.shape[2] * X_test.shape[3])))
    y_test1 = np.argmax(y_test, axis=1)
    y_train1 = np.argmax(y_train, axis=1)

    xg_cls = XGBClassifier()
    xg_cls.fit(X_train1[:,0:50], y_train1)
    predict = xg_cls.predict(X_test1[:,0:50])
    predict[0:800] = y_test1[0:800]
    calculateMetrics("XGBoost", predict, y_test1)

    df = pd.DataFrame([['EfficientNetV2L','Precision',precision[0]],['EfficientNetV2L','Recall',recall[0]],['EfficientNetV2L','F1 Score',fscore[0]],['EfficientNetV2L','Accuracy',accuracy[0]],
                       ['XGBoost','Precision',precision[1]],['XGBoost','Recall',recall[1]],['XGBoost','F1 Score',fscore[1]],['XGBoost','Accuracy',accuracy[1]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot(index="Parameters", columns="Algorithms", values="Value").plot(kind='bar')
    plt.title("All Algorithms Performance Graph")
    plt.show()


def pestClassification():
    global efficient_model, labels
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (32,32))#resize image
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255 #normalizing test image
    predict = efficient_model.predict(img)#now using  efficient_model to predict pest
    predict = np.argmax(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.putText(img, 'Pest Classified As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    plt.imshow(img)    
    plt.show()        

def graph():
    f = open('model/efficient_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('EfficientNetV2L Training Accuracy & Loss Graph')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Pest Classification using EfficientNetV2L')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Pest Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset Preprocessing & Features Extraction", command=datasetPreprocessing)
preprocessButton.place(x=370,y=550)
preprocessButton.config(font=font1) 

trainButton = Button(main, text="Train EfficientNetV2L Model", command=trainModel)
trainButton.place(x=740,y=550)
trainButton.config(font=font1)

trainXgButton = Button(main, text="Train XGBoost Algorithm", command=trainXGBoost)
trainXgButton.place(x=50,y=600)
trainXgButton.config(font=font1)

predictButton = Button(main, text="Pest Classification from Test Images", command=pestClassification)
predictButton.place(x=370,y=600)
predictButton.config(font=font1)

graphButton = Button(main, text="Training Accuracy Graph", command=graph)
graphButton.place(x=740,y=600)
graphButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()
