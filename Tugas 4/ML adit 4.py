import csv
import random
import math

# Check the versions of libraries
 
# Python version
import sys
 # print('Python: {}'.format(sys.version))
 # scipy
import scipy
 # print('scipy: {}'.format(scipy.__version__))
 # numpy
import numpy 
 # print('numpy: {}'.format(numpy.__version__))
 # matplotlib
import matplotlib
 # print('matplotlib: {}'.format(matplotlib.__version__))
 # pandas
import pandas
 # print('pandas: {}'.format(pandas.__version__))
 # scikit-learn
import sklearn
 # print('sklearn: {}'.format(sklearn.__version__))


 # Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

with open('C:/Users/adit/Downloads/iris.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     spamreader = list(spamreader)     

 # print(spamreader[1][5])     


indextest = []
indexBaru = []
index = list(range(0,150))
indextest.extend(list(range(0,10)))
indextest.extend(list(range(50,60)))
indextest.extend(list(range(100,110)))
 # print("Test Data : ", indextest)
indexBaru.append(0)
indexBaru=[x for x in index if x not in indextest]
 # print("Train Data : ", indexBaru)

AEA = []
AAAT = []
AEAT = []
AAA = []

target = []
target2 = []
target_Output = []
target_Output_2 = []
sigmoid_Output = []
sigmoid_Output_2 = []
sigmoid = []
sigmoid2 = []
prediction = []
prediction2 = []
error = []
error2 = []
D_theta1 = []
D_theta2 = []
D_theta3 = []
D_theta4 = []
D_theta5 = []
D_theta6 = []
D_theta7 = []
D_theta8 = []
D_theta_Hidden_1 = []
D_theta_Hidden_2 = []
D_theta_Hidden_3 = []
D_theta_Hidden_4 = []
D_bias = []
D_bias2 = []
D_bias_Hidden_1 = []
D_bias_Hidden_2 = []
lamda1 = []
lamda2 = []

targetTest = []
target2Test = []
target_Output_Test = []
target_Output_2_Test = []
sigmoidTest = []
sigmoid2Test = []
sigmoid_Output_Test = []
sigmoid_Output_2_Test = []
predictionTest = []
prediction2Test = []
errorTest = []
error2Test = []


Theta_1=random.random()
Theta_2=random.random()
Theta_3=random.random()
Theta_4=random.random()
Theta_5=random.random()
Theta_6=random.random()
Theta_7=random.random()
Theta_8=random.random()
Theta_Hidden_1=random.random()
Theta_Hidden_2=random.random()
Theta_Hidden_3=random.random()
Theta_Hidden_4=random.random()

Theta = [[Theta_1, Theta_2],[Theta_3,Theta_4],[Theta_5, Theta_6],[Theta_7, Theta_8]]
Theta_Hidden = [[Theta_Hidden_1, Theta_Hidden_2],[Theta_Hidden_3, Theta_Hidden_4]]

Bias=random.random()
Bias_2=random.random()
Bias_Hidden=random.random()
Bias_Hidden_2=random.random()

accuracy_avg = 0
totalError = 0
averageError = 0 
averageErrorArray = []
averageAccuracyArray = []
accuracy_avgTest = 0
totalErrorTest = 0
averageErrorTest = 0 
averageErrorArrayTest = []
averageAccuracyArrayTest = []
epo = []

for epoch in range (1,201):
    print("Epoch : ", epoch)
    print("------------------------------------------------------------------------------------")   
    accuracy = 0
    error_array = [0]
    error_array_test = [0]
    accuracyTest = 0
    print("TRAIN : ")
    for i in range(len(indexBaru)):
        if(epoch==1):
	        target.append(0),target2.append(0),sigmoid.append(0),sigmoid2.append(0),target_Output.append(0),target_Output_2.append(0),sigmoid_Output.append(0),sigmoid_Output_2.append(0),prediction.append(0),prediction2.append(0),error.append(0),error2.append(0)
	        D_theta1.append(0),D_theta2.append(0),D_theta3.append(0),D_theta4.append(0),D_theta5.append(0),D_theta6.append(0),D_theta7.append(0),D_theta8.append(0),
	        D_theta_Hidden_1.append(0),D_theta_Hidden_2.append(0),D_theta_Hidden_3.append(0),D_theta_Hidden_4.append(0),D_bias_Hidden_1.append(0),D_bias_Hidden_2.append(0),D_bias.append(0),D_bias2.append(0),lamda1.append(0),lamda2.append(0)

        target[i]=float(spamreader[indexBaru[i]][0])*Theta_1+float(spamreader[indexBaru[i]][1])*Theta_2+float(spamreader[indexBaru[i]][2])*Theta_3+float(spamreader[indexBaru[i]][3])*Theta_4+Bias
         # print("target1: %s" % (target[i]))
        
        target2[i]=float(spamreader[indexBaru[i]][0])*Theta_5+float(spamreader[indexBaru[i]][1])*Theta_6+float(spamreader[indexBaru[i]][2])*Theta_7+float(spamreader[indexBaru[i]][3])*Theta_8+Bias_2
         # print("target2: %s" % (target2[i]))

        sigmoid[i]=1/(1+math.exp(-target[i]))
         # print("sigmoid: %s" % (sigmoid[i]))

        sigmoid2[i]=1/(1+math.exp(-target2[i]))
         # print("sigmoid2: %s" % (sigmoid2[i]))

        target_Output[i]=float(sigmoid[i])*Theta_Hidden_1+float(sigmoid[i])*Theta_Hidden_2+Bias_Hidden
         # print("target1_Output: %s" % (target_Output[i]))

        target_Output_2[i]=float(sigmoid2[i])*Theta_Hidden_3+float(sigmoid2[i])*Theta_Hidden_4+Bias_Hidden_2
         # print("target2_Output: %s" % (target_Output_2[i]))	        

        sigmoid_Output[i]=1/(1+math.exp(-target_Output[i]))
         # print("sigmoid_Output: %s" % ( sigmoid_Output[i]))

        sigmoid_Output_2[i]=1/(1+math.exp(-target_Output_2[i]))
         # print("sigmoid_Output_2: %s" % ( sigmoid_Output_2[i]))

        if(sigmoid_Output[i]<float(0.5)):
            prediction[i]=0
        else:
            prediction[i]=1

        if(sigmoid_Output_2[i]<float(0.5)):
            prediction2[i]=0
        else:
            prediction2[i]=1
    
        # print("category:", (spamreader[i][5]))
        # print("category2:", (spamreader[i][6]))
        # print("prediction: %s" % (prediction[i]))
        # print("prediction2: %s" % (prediction2[i]))


        if prediction[i]==int(spamreader[indexBaru[i]][5]) and prediction2[i]==int(spamreader[indexBaru[i]][6]):
        	accuracy=accuracy+1
        	
         # print("Accuracy : ", accuracy)

        error[i]=(sigmoid_Output[i]-float(spamreader[indexBaru[i]][5]))**2
         # print("error: %s" % (error[i]))

        error2[i]=(sigmoid_Output_2[i]-float(spamreader[indexBaru[i]][6]))**2
         # print("error2: %s" % (error2[i]))
        
        error_array[0]+=(error[i]+error2[i])/2

        D_theta_Hidden_1[i]=2*(sigmoid_Output[i]-float(spamreader[indexBaru[i]][5]))*(1-sigmoid_Output[i])*sigmoid_Output[i]*sigmoid[i]
        # print("D_theta_Hidden_1: %s" % (D_theta_Hidden_1[i]))

        D_theta_Hidden_2[i]=2*(sigmoid_Output_2[i]-float(spamreader[indexBaru[i]][6]))*(1-sigmoid_Output_2[i])*sigmoid_Output_2[i]*sigmoid[i]
        # print("D_theta_Hidden_2: %s" % (D_theta_Hidden_2[i]))

        D_theta_Hidden_3[i]=2*(sigmoid_Output[i]-float(spamreader[indexBaru[i]][5]))*(1-sigmoid_Output[i])*sigmoid_Output[i]*sigmoid2[i]
        # print("D_theta_Hidden_3: %s" % (D_theta_Hidden_3[i]))

        D_theta_Hidden_4[i]=2*(sigmoid_Output_2[i]-float(spamreader[indexBaru[i]][6]))*(1-sigmoid_Output_2[i])*sigmoid_Output_2[i]*sigmoid2[i]
        # print("D_theta_Hidden_4: %s" % (D_theta_Hidden_4[i]))

        D_bias_Hidden_1[i]=2*(sigmoid_Output[i]-float(spamreader[indexBaru[i]][5]))*(1-sigmoid_Output[i])*sigmoid_Output[i]*1
        # print("D_bias_Hidden_1: %s" % (D_bias_Hidden_1[i]))

        D_bias_Hidden_2[i]=2*(sigmoid_Output_2[i]-float(spamreader[indexBaru[i]][6]))*(1-sigmoid_Output_2[i])*sigmoid_Output_2[i]*1
        # print("D_bias_Hidden_2: %s" % (D_bias_Hidden_2[i]))
        
        lamda1[i] = 2*(sigmoid_Output[i]-float(spamreader[indexBaru[i]][5]))*(1-sigmoid_Output[i])*sigmoid_Output[i]

        lamda2[i] = 2*(sigmoid_Output_2[i]-float(spamreader[indexBaru[i]][6]))*(1-sigmoid_Output_2[i])*sigmoid_Output_2[i]

        D_theta1[i]= (lamda1[i]*Theta_Hidden_1+lamda2[i]*Theta_Hidden_3)*sigmoid[i]*(1-sigmoid[i])*float(spamreader[indexBaru[i]][0])
        # print("D_theta1: %s" % (D_theta1[i]))
        D_theta2[i]= (lamda1[i]*Theta_Hidden_1+lamda2[i]*Theta_Hidden_3)*sigmoid[i]*(1-sigmoid[i])*float(spamreader[indexBaru[i]][1])
        # print("D_theta2: %s" % (D_theta2[i]))
        D_theta3[i]= (lamda1[i]*Theta_Hidden_1+lamda2[i]*Theta_Hidden_3)*sigmoid[i]*(1-sigmoid[i])*float(spamreader[indexBaru[i]][2])
        # print("D_theta3: %s" % (D_theta3[i]))
        D_theta4[i]= (lamda1[i]*Theta_Hidden_1+lamda2[i]*Theta_Hidden_3)*sigmoid[i]*(1-sigmoid[i])*float(spamreader[indexBaru[i]][3])
        # print("D_theta4: %s" % (D_theta4[i]))
        D_theta5[i]= (lamda1[i]*Theta_Hidden_2+lamda2[i]*Theta_Hidden_4)*sigmoid2[i]*(1-sigmoid2[i])*float(spamreader[indexBaru[i]][0])
        # print("D_theta5: %s" % (D_theta5[i]))
        D_theta6[i]= (lamda1[i]*Theta_Hidden_2+lamda2[i]*Theta_Hidden_4)*sigmoid2[i]*(1-sigmoid2[i])*float(spamreader[indexBaru[i]][1])
        # print("D_theta6: %s" % (D_theta6))
        D_theta7[i]= (lamda1[i]*Theta_Hidden_2+lamda2[i]*Theta_Hidden_4)*sigmoid2[i]*(1-sigmoid2[i])*float(spamreader[indexBaru[i]][2])
        # print("D_theta7: %s" % (D_theta7[i]))
        D_theta8[i]= (lamda1[i]*Theta_Hidden_2+lamda2[i]*Theta_Hidden_4)*sigmoid2[i]*(1-sigmoid2[i])*float(spamreader[indexBaru[i]][3])
        # print("D_theta8: %s" % (D_theta8[i]))

        D_bias[i]= (lamda1[i]*Theta_Hidden_1+lamda2[i]*Theta_Hidden_3)*sigmoid[i]*(1-sigmoid[i])*1
        # print("D_theta_Hidden_1: %s" % (D_theta_Hidden_1[i]))
        D_bias2[i]= (lamda1[i]*Theta_Hidden_2+lamda2[i]*Theta_Hidden_4)*sigmoid2[i]*(1-sigmoid[i])*1
        # print("D_theta_Hidden_1: %s" % (D_theta_Hidden_1[i]))


        Theta_Hidden_1 = Theta_Hidden_1-(0.1*D_theta_Hidden_1[i])
        Theta_Hidden_2 = Theta_Hidden_2-(0.1*D_theta_Hidden_2[i])
        Theta_Hidden_3 = Theta_Hidden_3-(0.1*D_theta_Hidden_3[i])
        Theta_Hidden_4 = Theta_Hidden_4-(0.1*D_theta_Hidden_4[i])

        Bias_Hidden = Bias_Hidden-(0.1*D_bias_Hidden_1[i])
        Bias_Hidden_2 = Bias_Hidden_2-(0.1*D_bias_Hidden_2[i])

        Bias = Bias-(0.1*D_bias[i])
        Bias_2 = Bias_2-(0.1*D_bias2[i])

        Theta_1 = Theta_1-(0.1*D_theta1[i])
        Theta_2 = Theta_2-(0.1*D_theta2[i])
        Theta_3 = Theta_3-(0.1*D_theta3[i])
        Theta_4 = Theta_4-(0.1*D_theta4[i])
        Theta_5 = Theta_5-(0.1*D_theta5[i])
        Theta_6 = Theta_6-(0.1*D_theta6[i])
        Theta_7 = Theta_7-(0.1*D_theta7[i])
        Theta_8 = Theta_8-(0.1*D_theta8[i])

    AEA.append(error_array[0]/len(indexBaru))
    AAA.append(accuracy/len(indexBaru))
    print("Ërror Train : ", error_array[0]/len(indexBaru) )
    print("Accuracy Train : ", accuracy/len(indexBaru))
    print("---------------------------------------------------")
    print("TEST")
    for i in range(len(indextest)):
        if(epoch==1):
	        targetTest.append(0),target2Test.append(0),sigmoidTest.append(0),sigmoid2Test.append(0),target_Output_Test.append(0),target_Output_2_Test.append(0),sigmoid_Output_Test.append(0),sigmoid_Output_2_Test.append(0),predictionTest.append(0),prediction2Test.append(0),errorTest.append(0),error2Test.append(0)

        targetTest[i]=float(spamreader[indextest[i]][0])*Theta_1+float(spamreader[indextest[i]][1])*Theta_2+float(spamreader[indextest[i]][2])*Theta_3+float(spamreader[indextest[i]][3])*Theta_4+Bias
         # print("target1Test: %s" % (targetTest[i]))
        
        target2Test[i]=float(spamreader[indextest[i]][0])*Theta_5+float(spamreader[indextest[i]][1])*Theta_6+float(spamreader[indextest[i]][2])*Theta_7+float(spamreader[indextest[i]][3])*Theta_8+Bias_2
         # print("target2Test: %s" % (target2Test[i]))

        sigmoidTest[i]=1/(1+math.exp(-targetTest[i]))
         # print("sigmoidTest: %s" % (sigmoidTest[i]))

        sigmoid2Test[i]=1/(1+math.exp(-target2Test[i]))
         # print("sigmoid2Test: %s" % (sigmoid2Test[i]))

        target_Output_Test[i]=float(sigmoidTest[i])*Theta_Hidden_1+float(sigmoidTest[i])*Theta_Hidden_2+Bias_Hidden
         # print("target1_Output_Test: %s" % (target_Output_Test[i]))

        target_Output_2_Test[i]=float(sigmoid2Test[i])*Theta_Hidden_3+float(sigmoid2Test[i])*Theta_Hidden_4+Bias_Hidden_2
         # print("target2_Output_Test: %s" % (target_Output_2_Test[i]))	        

        sigmoid_Output_Test[i]=1/(1+math.exp(-target_Output_Test[i]))
         # print("sigmoid_Output_Test: %s" % ( sigmoid_Output_Test[i]))

        sigmoid_Output_2_Test[i]=1/(1+math.exp(-target_Output_2_Test[i]))
         # print("sigmoid_Output_2_Test: %s" % ( sigmoid_Output_2_Test[i]))

        if(sigmoid_Output_Test[i]<float(0.5)):
            predictionTest[i]=0
        else:
            predictionTest[i]=1

        if(sigmoid_Output_2_Test[i]<float(0.5)):
            prediction2Test[i]=0
        else:
            prediction2Test[i]=1
    
        # print("category:", (spamreader[indextest[i]][5]))
        # print("category2:", (spamreader[indextest[i]][6]))
        # print("prediction: %s" % (predictionTest[i]))
        # print("prediction2: %s" % (prediction2Test[i]))


        if predictionTest[i]==int(spamreader[indextest[i]][5]) and prediction2Test[i]==int(spamreader[indextest[i]][6]):
        	accuracyTest=accuracyTest+1
        	
         # print("accuracyTest : ", accuracyTest)

        errorTest[i]=(sigmoid_Output_Test[i]-float(spamreader[indextest[i]][5]))**2
         # print("errorTest: %s" % (errorTest[i]))

        error2Test[i]=(sigmoid_Output_2_Test[i]-float(spamreader[indextest[i]][6]))**2
         # print("error2Test: %s" % (error2Test[i]))
        
        error_array_test[0]+=(errorTest[i]+error2Test[i])/2
    AEAT.append(error_array_test[0]/len(indextest))
    AAAT.append(accuracyTest/len(indextest))
    print("Ërror Test : ",  error_array_test[0]/len(indextest))
    print("Accuracy Test : ", accuracyTest/len(indextest))
    epo.append(epoch)
    print("---------------------------------------------------")

plt.figure(1)
plt.plot(epo, AEA, label='ErrorTrain')
plt.plot(epo, AEAT, label='errorTest')
plt.title('plot error')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error1')

plt.figure(2)
plt.plot(epo, AAA, label='AccuracyTrain')
plt.plot(epo, AAAT, label='accuracyTest')
plt.title('plot accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy1')

plt.show()


