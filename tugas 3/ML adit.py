import csv
import random
import math

# Check the versions of libraries
 
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy 
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


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

print(spamreader[1][5])     


indextest = []
indexBaru = []
index = list(range(0,150))
for i in range(0,5):
    indextest.append(list(range(10*i,10*(i+1))))
    indextest[i].extend(list(range(50+10*i,60+10*i)))
    indextest[i].extend(list(range(100+10*i,110+10*i)))
    print("Test",i+1," ", indextest[i])
    indexBaru.append(0)
    indexBaru[i]=[x for x in index if x not in indextest[i]]
    print("Train",i+1," ", indexBaru[i])

AEA = []
AAAT = []
AEAT = []
AAA = []

for n in range(0,5): 
	target = []
	target2 = []
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
	D_bias = []
	D_bias2 = []

	targetTest = []
	target2Test = []
	sigmoidTest = []
	sigmoid2Test = []
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
	Bias=random.random()
	Bias_2=random.random()
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

	for epoch in range (1,100):
	    print("Epoch : ", epoch)
	    #print("------------------------------------------------------------------------------------")   
	    accuracy = 0 
	    for i in range(len(indexBaru[n])):
	        #print("Iteration : ",i+1)
	        target.append(0),target2.append(0),sigmoid.append(0),sigmoid2.append(0),prediction.append(0),prediction2.append(0),error.append(0),error2.append(0)
	        D_theta1.append(0),D_theta2.append(0),D_theta3.append(0),D_theta4.append(0),D_theta5.append(0),D_theta6.append(0),D_theta7.append(0),D_theta8.append(0)
	        D_bias.append(0),D_bias2.append(0)

	        target[i]=float(spamreader[indexBaru[n][i]][0])*Theta_1+float(spamreader[indexBaru[n][i]][1])*Theta_2+float(spamreader[indexBaru[n][i]][2])*Theta_3+float(spamreader[indexBaru[n][i]][3])*Theta_4+Bias
	        #print("target1: %s" % (target[i]))
	        
	        target2[i]=float(spamreader[indexBaru[n][i]][0])*Theta_5+float(spamreader[indexBaru[n][i]][1])*Theta_6+float(spamreader[indexBaru[n][i]][2])*Theta_7+float(spamreader[indexBaru[n][i]][3])*Theta_8+Bias_2
	        #print("target2: %s" % (target2[i]))

	        sigmoid[i]=1/(1+math.exp(-target[i]))
	        #print("sigmoid: %s" % (sigmoid[i]))

	        sigmoid2[i]=1/(1+math.exp(-target2[i]))
	        #print("sigmoid2: %s" % (sigmoid2[i]))

	        if(sigmoid[i]<float(0.5)):
	            prediction[i]=0
	        else:
	            prediction[i]=1

	        if(sigmoid2[i]<float(0.5)):
	            prediction2[i]=0
	        else:
	            prediction2[i]=1
	        
	        # print("category:", (spamreader[i][5]))
	        # print("category:", (spamreader[i][6]))
	        # print("prediction: %s" % (prediction[i]))
	        # print("prediction2: %s" % (prediction2[i]))


	        if prediction[i]==int(spamreader[indexBaru[n][i]][5]) and prediction2[i]==int(spamreader[indexBaru[n][i]][6]):
	        	accuracy=accuracy+1
	        	
	        #print("Accuracy : ", accuracy)

	        error[i]=(sigmoid[i]-float(spamreader[indexBaru[n][i]][5]))**2
	        #print("error: %s" % (error[i]))

	        error2[i]=(sigmoid2[i]-float(spamreader[indexBaru[n][i]][6]))**2
	        #print("error2: %s" % (error2[i]))
	        
	        totalError=(error[i]+error2[i])/2

	        D_theta1[i]=2*(sigmoid[i]-float(spamreader[indexBaru[n][i]][5]))*(1-sigmoid[i])*sigmoid[i]*float(spamreader[indexBaru[n][i]][0])
	        #print("D_theta_1: %s" % (D_theta1[i]))

	        D_theta2[i]=2*(sigmoid[i]-float(spamreader[indexBaru[n][i]][5]))*(1-sigmoid[i])*sigmoid[i]*float(spamreader[indexBaru[n][i]][1])
	        #print("D_theta_2: %s" % (D_theta2[i]))

	        D_theta3[i]=2*(sigmoid[i]-float(spamreader[indexBaru[n][i]][5]))*(1-sigmoid[i])*sigmoid[i]*float(spamreader[indexBaru[n][i]][2])
	        #print("D_theta_3: %s" % (D_theta3[i]))

	        D_theta4[i]=2*(sigmoid[i]-float(spamreader[indexBaru[n][i]][5]))*(1-sigmoid[i])*sigmoid[i]*float(spamreader[indexBaru[n][i]][3])
	        #print("D_theta_4: %s" % (D_theta4[i]))

	        D_theta5[i]=2*(sigmoid2[i]-float(spamreader[indexBaru[n][i]][6]))*(1-sigmoid2[i])*sigmoid2[i]*float(spamreader[indexBaru[n][i]][0])
	        #print("D_theta_5: %s" % (D_theta5[i]))

	        D_theta6[i]=2*(sigmoid2[i]-float(spamreader[indexBaru[n][i]][6]))*(1-sigmoid2[i])*sigmoid2[i]*float(spamreader[indexBaru[n][i]][1])
	        #print("D_theta_6: %s" % (D_theta6[i]))

	        D_theta7[i]=2*(sigmoid2[i]-float(spamreader[indexBaru[n][i]][6]))*(1-sigmoid2[i])*sigmoid2[i]*float(spamreader[indexBaru[n][i]][2])
	        #print("D_theta_7: %s" % (D_theta7[i]))

	        D_theta8[i]=2*(sigmoid2[i]-float(spamreader[indexBaru[n][i]][6]))*(1-sigmoid2[i])*sigmoid2[i]*float(spamreader[indexBaru[n][i]][3])
	        #print("D_theta_8: %s" % (D_theta8[i]))

	        D_bias[i]=2*(sigmoid[i]-float(spamreader[indexBaru[n][i]][5]))*(1-sigmoid[i])*sigmoid[i]*1
	        #print("D_bias: %s" % (D_bias[i]))

	        D_bias2[i]=2*(sigmoid2[i]-float(spamreader[indexBaru[n][i]][6]))*(1-sigmoid2[i])*sigmoid2[i]*1
	        #print("D_bias_2: %s" % (D_bias2[i]))

	        Theta_1=Theta_1-(0.1*D_theta1[i])
	        #print("Theta_1: %s" % (Theta_1))

	        Theta_2=Theta_2-(0.1*D_theta2[i])
	        #print("Theta_2: %s" % (Theta_2))

	        Theta_3=Theta_3-(0.1*D_theta3[i])
	        #print("Theta_3: %s" % (Theta_3))

	        Theta_4=Theta_4-(0.1*D_theta4[i])
	        #print("Theta_4: %s" % (Theta_4))

	        Theta_5=Theta_5-(0.1*D_theta5[i])
	        #print("Theta_5: %s" % (Theta_5))

	        Theta_6=Theta_6-(0.1*D_theta6[i])
	        #print("Theta_6: %s" % (Theta_6))

	        Theta_7=Theta_7-(0.1*D_theta7[i])
	        #print("Theta_7: %s" % (Theta_7))

	        Theta_8=Theta_8-(0.1*D_theta8[i])
	        #print("Theta_8: %s" % (Theta_8))

	        BiasB=Bias-(0.1*D_bias[i])
	        #print("Bias: %s" % (Bias))

	        Bias_2=Bias_2-(0.1*D_bias2[i])
	        #print("Bias2: %s" % (Bias_2))
	        

	        #print("--------------------------------------------------------")
	    accuracy_avg = (accuracy/120)
	    averageError = (totalError/120)
	    
	    averageAccuracyArray.append(accuracy_avg)
	    averageErrorArray.append(averageError)
	    epo.append(epoch)

	    print("Accuracy AVG : ", accuracy_avg)
	    print("Error AVG : ", averageError)

	    #print("AVG Accuracy Array : ", averageAccuracyArray)
	    #print("AVG Error Array : ", averageErrorArray)
	    accuracyTest = 0

	    for i in range(len(indextest[n])):
	        #print("Iteration : ",i+1)
	        targetTest.append(0),target2Test.append(0),sigmoidTest.append(0),sigmoid2Test.append(0),predictionTest.append(0),prediction2Test.append(0),errorTest.append(0),error2Test.append(0)
	   

	        targetTest[i]=float(spamreader[indextest[n][i]][0])*Theta_1+float(spamreader[indextest[n][i]][1])*Theta_2+float(spamreader[indextest[n][i]][2])*Theta_3+float(spamreader[indextest[n][i]][3])*Theta_4+Bias
	        #print("target1: %s" % (target[i]))
	        
	        target2Test[i]=float(spamreader[indextest[n][i]][0])*Theta_5+float(spamreader[indextest[n][i]][1])*Theta_6+float(spamreader[indextest[n][i]][2])*Theta_7+float(spamreader[indextest[n][i]][3])*Theta_8+Bias_2
	        #print("target2: %s" % (target2[i]))

	        sigmoidTest[i]=1/(1+math.exp(-targetTest[i]))
	        #print("sigmoid: %s" % (sigmoid[i]))

	        sigmoid2Test[i]=1/(1+math.exp(-target2Test[i]))
	        #print("sigmoid2: %s" % (sigmoid2[i]))

	        if(sigmoidTest[i]<float(0.5)):
	            predictionTest[i]=0
	        else:
	            predictionTest[i]=1

	        if(sigmoid2Test[i]<float(0.5)):
	            prediction2Test[i]=0
	        else:
	            prediction2Test[i]=1
	        
	        # print("category:", (spamreader[i][5]))
	        # print("category:", (spamreader[i][6]))
	        # print("prediction: %s" % (prediction[i]))
	        # print("prediction2: %s" % (prediction2[i]))


	        if predictionTest[i]==int(spamreader[indextest[n][i]][5]) and prediction2[i]==int(spamreader[indextest[n][i]][6]):
	        	accuracyTest=accuracyTest+1
	        	
	        #print("Accuracy : ", accuracyTest)

	        errorTest[i]=(sigmoidTest[i]-float(spamreader[indextest[n][i]][5]))**2
	        #print("error: %s" % (errorTest[i]))

	        error2Test[i]=(sigmoid2Test[i]-float(spamreader[indextest[n][i]][6]))**2
	        #print("error2: %s" % (error2Test[i]))
	        
	        totalErrorTest=(errorTest[i]+error2Test[i])/2 


	    accuracy_avgTest = (accuracyTest/120)
	    averageErrorTest = (totalErrorTest/120)
	    
	    averageAccuracyArrayTest.append(accuracy_avgTest)
	    averageErrorArrayTest.append(averageErrorTest)

	    print("Accuracy AVG Test: ", accuracy_avgTest)
	    print("Error AVG Test: ", averageErrorTest)

	    #print("AVG Accuracy Array Test : ", averageAccuracyArrayTest)
	    #print("AVG Error Array Test: ", averageErrorArrayTest)

	AAA.append(averageAccuracyArray)
	AEA.append(averageErrorArray)
	AAAT.append(averageAccuracyArrayTest)
	AEAT.append(averageErrorArrayTest)

for i in range(0,5):
	print("AEA ", i ," : ", AEA[i])
plt.figure(1)
plt.plot(epo, AEA[0], label='ErrorTrain')
plt.plot(epo, AEAT[0], label='errorTest')
plt.title('plot error')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error1')

plt.figure(2)
plt.plot(epo, AAA[0], label='AccuracyTrain')
plt.plot(epo, AAAT[0], label='accuracyTest')
plt.title('plot accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy1')

plt.figure(3)
plt.plot(epo, AEA[1], label='ErrorTrain')
plt.plot(epo, AEAT[1], label='errorTest')
plt.title('plot error')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error1')

plt.figure(4)
plt.plot(epo, AAA[1], label='AccuracyTrain')
plt.plot(epo, AAAT[1], label='accuracyTest')
plt.title('plot accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy1')


plt.figure(5)
plt.plot(epo, AEA[2], label='ErrorTrain')
plt.plot(epo, AEAT[2], label='errorTest')
plt.title('plot error')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error1')

plt.figure(6)
plt.plot(epo, AAA[2], label='AccuracyTrain')
plt.plot(epo, AAAT[2], label='accuracyTest')
plt.title('plot accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy1')


plt.figure(7)
plt.plot(epo, AEA[3], label='ErrorTrain')
plt.plot(epo, AEAT[3], label='errorTest')
plt.title('plot error')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error1')

plt.figure(8)
plt.plot(epo, AAA[3], label='AccuracyTrain')
plt.plot(epo, AAAT[3], label='accuracyTest')
plt.title('plot accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy1')


plt.figure(9)
plt.plot(epo, AEA[4], label='ErrorTrain')
plt.plot(epo, AEAT[4], label='errorTest')
plt.title('plot error')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error1')

plt.figure(10)
plt.plot(epo, AAA[4], label='AccuracyTrain')
plt.plot(epo, AAAT[4], label='accuracyTest')
plt.title('plot accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy1')

plt.figure(11)
plt.plot(epo, (numpy.array(AEA[0])+numpy.array(AEA[1])+numpy.array(AEA[2])+numpy.array(AEA[3])+numpy.array(AEA[4]))/5, label='ErrorTrain')
plt.plot(epo, (numpy.array(AEAT[0])+numpy.array(AEAT[1])+numpy.array(AEAT[2])+numpy.array(AEAT[3])+numpy.array(AEAT[4]))/5, label='errorTest')
plt.title('plot error')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error1')

plt.figure(12)
plt.plot(epo, (numpy.array(AAA[0])+numpy.array(AAA[1])+numpy.array(AAA[2])+numpy.array(AAA[3])+numpy.array(AAA[4]))/5, label='AccuracyTrain')
plt.plot(epo, (numpy.array(AAAT[0])+numpy.array(AAAT[1])+numpy.array(AAAT[2])+numpy.array(AAAT[3])+numpy.array(AAAT[4]))/5, label='accuracyTest')
plt.title('plot accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy1')

plt.show()


