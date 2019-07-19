from tkinter import *
from tkinter.colorchooser import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np 

Labels={'red-ish':0,'green-ish':1,'blue-ish':2,'orange-ish':3,'yellow-ish':4,'pink-ish':5,'purple-ish':6,'brown-ish':7,'grey-ish':8}
Label=['red-ish','green-ish','blue-ish','orange-ish','yellow-ish','pink-ish','purple-ish','brown-ish','grey-ish']

dataset = pd.read_csv('ColorDataset.csv')
#Preprocessing of Data
#Label Mapping 
dataset['label']=dataset['label'].map(Labels).to_frame()

#handling null values
dataset['label'].fillna('grey-ish', inplace = True)
dataset['r'].fillna(0, inplace = True)
dataset['g'].fillna(0, inplace = True)
dataset['b'].fillna(0, inplace = True)

#seperating columns 

X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values

ndataset=dataset[['r','g','b']]

#Box Plot to check outliers
import seaborn as sns
sns.boxplot(data=ndataset, orient="h")
plt.show()

#train and test data split

train,test,train_op,test_op=train_test_split(X,y,test_size=0.20)
train_op=train_op.astype('int')

#Model Initializations
#initalization of naive bayes classifier
gnb=GaussianNB()

#initalization of decision tree classifier
clf=tree.DecisionTreeClassifier()

#initalization of K neighbours classifier
neigh = KNeighborsClassifier(n_neighbors=70)

#training all 3 models
gnb.fit(train,train_op)
clf.fit(train,train_op)
neigh.fit(train,train_op )

global in_color
def getColor():
    in_color = askcolor()
    color=[]
    for i in in_color[0]:
         color.append(int(i))
    color=[color]
    index=int(gnb.predict(np.array(color)))

    print ("\n**********Naive Bayes Classifier**********\n")
    print ("Prediction of color by naive bayes classifier is : "+Label[index])
    
    pred=gnb.predict(train)
    score=accuracy_score(train_op,pred)
    print ("\nAccuracy of naive bayes classifier over train data is : "+str(score*100))
    print ("\nError Rate of naive bayes classifier over train data is : "+str(100-(score*100)))
    pred=gnb.predict(test)
    score=accuracy_score(test_op,pred)
    print ("\nAccuracy of naive bayes classifier over test data is : "+str(score*100))
    print ("\nError Rate of naive bayes classifier over test data is : "+str(100-(score*100)))
    scores = cross_val_score(gnb, test, test_op, cv=10)
    print ("\nCross Validation Accuracy Score is :"+str(scores.mean()*100))
    print ("")
    

    matrix=confusion_matrix(test_op, pred)
    print (matrix)
    print ("\nPrecision Values")
    for i in range(9):
        tp=matrix[i][i]
        sums=0
        for j in range(9):
            sums+=matrix[j][i]
        precision=tp/sums
        print (Label[i]+":"+str(precision))
    print ("\nRecall Values")
    for i in range(9):
        tp=matrix[i][i]
        sums=0
        for j in range(9):
            sums+=matrix[i][j]
        Recall=tp/sums
        print (Label[i]+":"+str(Recall))
    
    
    print ("\n**********Decision Tree Classifier**********\n")
    
    index=int(clf.predict(np.array(color)))
    print ("Prediction of color by decision tree classifier is : "+Label[index])
    pred=clf.predict(train)
    score=accuracy_score(train_op,pred)
    print ("\nAccuracy of decision tree classifier over train data is : "+str(score*100))
    print ("\nError Rate of decision tree classifier over train data is : "+str(100-(score*100)))
   
    pred=clf.predict(test)
    score=accuracy_score(test_op,pred)
    print ("\nAccuracy of decision tree classifier over test data is : "+str(score*100))
    print ("\nError Rate of decision tree  classifier over test data is : "+str(100-(score*100)))
    dscores = cross_val_score(clf, test, test_op, cv=10)
    print ("\nCross Validation Accuracy Score is :"+str(dscores.mean()*100))
    print ("")

    matrix=confusion_matrix(test_op, pred)
    print (matrix)
    print ("\nPrecision Values")
    for i in range(9):
        tp=matrix[i][i]
        sums=0
        for j in range(9):
            sums+=matrix[j][i]
        precision=tp/sums
        print (Label[i]+":"+str(precision))
    print ("\nRecall Values")
    for i in range(9):
        tp=matrix[i][i]
        sums=0
        for j in range(9):
            sums+=matrix[i][j]
        Recall=tp/sums
        print (Label[i]+":"+str(Recall))

    print ("\n**********K Neighbours Classifier**********\n")

    index=int(neigh.predict(np.array(color)))
    print ("Prediction of color by K neighbours classifier is : "+ Label[index])

    pred=neigh.predict(train)
    score=accuracy_score(train_op,pred)
    print ("\nAccuracy of K neighbours classifier over train data is : "+ str(score*100))
    print ("\nError Rate of K neighbours classifier over train data is: "+str(100-(score*100)))

    pred=neigh.predict(test)
    score=accuracy_score(test_op,pred)
    print ("\nAccuracy of K neighbours classifier over test data is : "+ str(score*100))
    print ("\nError Rate of K neighbours classifier over test data is: "+str(100-(score*100)))
    kscores = cross_val_score(neigh, test, test_op, cv=10)
    print ("\nCross Validation Accuracy Score is :"+str(kscores.mean()*100))
    print ("")

    matrix=confusion_matrix(test_op, pred)
    print (matrix)
    print ("\nPrecision Values")
    for i in range(9):
        tp=matrix[i][i]
        sums=0
        for j in range(9):
            sums+=matrix[j][i]
        precision=tp/sums
        print (Label[i]+":"+str(precision))
    print ("\nRecall Values")
    for i in range(9):
        tp=matrix[i][i]
        sums=0
        for j in range(9):
            sums+=matrix[i][j]
        Recall=tp/sums
        print (Label[i]+":"+str(Recall))
   
    #plotting acurracy folds
    plt.plot(scores,"-p",label="Naive Bayes")
    plt.plot(dscores,"-p",label="Decision tree")
    plt.plot(kscores,"-p",label="KNN")
    plt.title("Cross Validation Scores")
    plt.xlabel("fold number")
    plt.ylabel("Accuracy score")
    plt.legend(loc='upper right')
    plt.show()
    
    





Button(text='Select Color want to classify', command=getColor).pack()

mainloop()
