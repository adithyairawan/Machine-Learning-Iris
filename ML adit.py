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

with open('C:/Users/adit/Downloads/iris.csv', newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     spamreader = list(spamreader)
     
print(spamreader[1][5])     


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
print("Bias : %s" % Bias)
print("Bias_2: %s" % Bias_2)

class target:
    def __init__(self, Theta_1, Theta_2,Theta_3,Theta_4,spamreader,Bias):
      self.Theta_1 = Theta_1
      self.Theta_2 = Theta_2
      self.Theta_3 = Theta_3
      self.Theta_4 = Theta_4
      self.spamreader = spamreader

target = []
i=0
for i in range(149):
 target.append(i)
 target[i]=float(spamreader[i][0])*Theta_1+float(spamreader[i][1])*Theta_2+float(spamreader[i][2])*Theta_3+float(spamreader[i][3])*Theta_4+Bias
 print("target1: %s" % (target[i]))
 i = i+1

class target2:
    def __init__(self, Theta_5, Theta_6,Theta_7,Theta_8,spamreader,Bias_2):
      self.Theta_5 = Theta_5
      self.Theta_6 = Theta_6
      self.Theta_7 = Theta_7
      self.Theta_8 = Theta_8
      self.spamreader = spamreader

target2 = []
i=0
for i in range(149):
 target2.append(i)
 target2[i]=float(spamreader[i][0])*Theta_5+float(spamreader[i][1])*Theta_6+float(spamreader[i][2])*Theta_7+float(spamreader[i][3])*Theta_8+Bias_2
 print("target2: %s" % (target2[i]))
 i = i+1

class sigmoid:
    def __init__(self, target):
      self.target1 = target1

sigmoid = []
i=0
for i in range(149):
 sigmoid.append(i)
 sigmoid[i]=1/(1+math.exp(-target[i]))
 print("sigmoid: %s" % (sigmoid[i]))
 i = i+1

class sigmoid_2:
    def __init__(self, target):
      self.target2 = target2

sigmoid2 = []
i=0
for i in range(149):
 sigmoid2.append(i)
 sigmoid2[i]=1/(1+math.exp(-target2[i]))
 print("sigmoid2: %s" % (sigmoid2[i]))
 i = i+1

class prediction:
    def __init__(self, sigmoid):
      self.sigmoid = sigmoid

prediction = []
i=0
for i in range(149):
 prediction.append(i)
 if(sigmoid[i]<float(0.5)):
  prediction[i]=0
  print("prediction: %s" % (prediction[i]))
 else:
  prediction[i]=1
  print("prediction: %s" % (prediction[i]))
 
 i = i+1

class prediction_2:
    def __init__(self, sigmoid):
      self.sigmoid2 = sigmoid2

prediction2 = []
i=0
for i in range(149):
 prediction2.append(i)
 if(sigmoid2[i]<float(0.5)):
  prediction2[i]=0
  print("prediction2: %s" % (prediction2[i]))
 else:
  prediction2[i]=1
  print("prediction2: %s" % (prediction2[i]))
 
 i = i+1

class error:
	"""docstring for error"""
	def __init__(self, sigmoid, spamreader):
		self.sigmoid = sigmoid
		self.spamreader= spamreader
i=0
error = []
for i in range(149):
 error.append(i)
 error[i]=(sigmoid[i]-float(spamreader[i][5]))*(sigmoid[i]-float(spamreader[i][5]))
 print("error: %s" % (error[i]))
 i=i+1
		
class error_2:
	"""docstring for error"""
	def __init__(self, sigmoid, spamreader):
		self.sigmoid2 = sigmoid2
		self.spamreader= spamreader
i=0
error2 = []
for i in range(149):
 error2.append(i)
 error2[i]=(sigmoid2[i]-float(spamreader[i][5]))*(sigmoid2[i]-float(spamreader[i][5]))
 print("error2: %s" % (error2[i]))
 i=i+1

class D_theta_1:
	"""docstring for error"""
	def __init__(self, sigmoid, spamreader):
		self.sigmoid = sigmoid
		self.spamreader= spamreader
i=0
D_theta1 = []
for i in range(149):
 D_theta1.append(i)
 D_theta1[i]=2*(sigmoid[i]-float(spamreader[i][5]))*(1-sigmoid[i])*sigmoid[i]*float(spamreader[i][0])
 print("D_theta_1: %s" % (D_theta1[i]))
 i=i+1

class D_theta_2:
	"""docstring for error"""
	def __init__(self, sigmoid, spamreader):
		self.sigmoid = sigmoid
		self.spamreader= spamreader
i=0
D_theta2 = []
for i in range(149):
 D_theta2.append(i)
 D_theta2[i]=2*(sigmoid[i]-float(spamreader[i][5]))*(1-sigmoid[i])*sigmoid[i]*float(spamreader[i][1])
 print("D_theta_2: %s" % (D_theta2[i]))
 i=i+1

class D_theta_3:
	"""docstring for error"""
	def __init__(self, sigmoid, spamreader):
		self.sigmoid = sigmoid
		self.spamreader= spamreader
i=0
D_theta3 = []
for i in range(149):
 D_theta3.append(i)
 D_theta3[i]=2*(sigmoid[i]-float(spamreader[i][5]))*(1-sigmoid[i])*sigmoid[i]*float(spamreader[i][2])
 print("D_theta_3: %s" % (D_theta3[i]))
 i=i+1

class D_theta_4:
	"""docstring for error"""
	def __init__(self, sigmoid, spamreader):
		self.sigmoid = sigmoid
		self.spamreader= spamreader
i=0
D_theta4 = []
for i in range(149):
 D_theta4.append(i)
 D_theta4[i]=2*(sigmoid[i]-float(spamreader[i][5]))*(1-sigmoid[i])*sigmoid[i]*float(spamreader[i][3])
 print("D_theta_4: %s" % (D_theta4[i]))
 i=i+1

class D_theta_5:
	"""docstring for error"""
	def __init__(self, sigmoid_2, spamreader):
		self.sigmoid2 = sigmoid2
		self.spamreader= spamreader
i=0
D_theta5 = []
for i in range(149):
 D_theta5.append(i)
 D_theta5[i]=2*(sigmoid2[i]-float(spamreader[i][6]))*(1-sigmoid2[i])*sigmoid2[i]*float(spamreader[i][0])
 print("D_theta_5: %s" % (D_theta5[i]))
 i=i+1

class D_theta_6:
	"""docstring for error"""
	def __init__(self, sigmoid_2, spamreader):
		self.sigmoid2 = sigmoid2
		self.spamreader= spamreader
i=0
D_theta6 = []
for i in range(149):
 D_theta6.append(i)
 D_theta6[i]=2*(sigmoid2[i]-float(spamreader[i][6]))*(1-sigmoid2[i])*sigmoid2[i]*float(spamreader[i][1])
 print("D_theta_6: %s" % (D_theta6[i]))
 i=i+1

class D_theta_7:
	"""docstring for error"""
	def __init__(self, sigmoid_2, spamreader):
		self.sigmoid2 = sigmoid2
		self.spamreader= spamreader
i=0
D_theta7 = []
for i in range(149):
 D_theta7.append(i)
 D_theta7[i]=2*(sigmoid2[i]-float(spamreader[i][6]))*(1-sigmoid2[i])*sigmoid2[i]*float(spamreader[i][2])
 print("D_theta_7: %s" % (D_theta7[i]))
 i=i+1

class D_theta_8:
	"""docstring for error"""
	def __init__(self, sigmoid_2, spamreader):
		self.sigmoid2 = sigmoid2
		self.spamreader= spamreader
i=0
D_theta8 = []
for i in range(149):
 D_theta8.append(i)
 D_theta8[i]=2*(sigmoid2[i]-float(spamreader[i][6]))*(1-sigmoid2[i])*sigmoid2[i]*float(spamreader[i][3])
 print("D_theta_8: %s" % (D_theta8[i]))
 i=i+1

class D_bias:
	"""docstring for error"""
	def __init__(self, sigmoid, spamreader):
		self.sigmoid = sigmoid
		self.spamreader= spamreader
i=0
D_bias = []
for i in range(149):
 D_bias.append(i)
 D_bias[i]=2*(sigmoid[i]-float(spamreader[i][5]))*(1-sigmoid[i])*sigmoid[i]*1
 print("D_bias: %s" % (D_bias[i]))
 i=i+1 

class D_bias_2:
	"""docstring for error"""
	def __init__(self, sigmoid, spamreader):
		self.sigmoid2 = sigmoid2
		self.spamreader= spamreader
i=0
D_bias2 = []
for i in range(149):
 D_bias2.append(i)
 D_bias2[i]=2*(sigmoid2[i]-float(spamreader[i][6]))*(1-sigmoid2[i])*sigmoid2[i]*1
 print("D_bias_2: %s" % (D_bias2[i]))
 i=i+1

class Theta_1_baru:
	"""docstring for error"""
	def __init__(self, Theta_1, D_theta1):
		self.Theta_1 = Theta_1
		self.D_theta1= D_theta1
i=0
Theta_1baru = []

for i in range(149):
 Theta_1baru.append(i)
 Theta_1baru[i]=Theta_1-(0.1*D_theta1[i])
 print("Theta_1_baru: %s" % (Theta_1baru[i]))
 i=i+1

class Theta_2_baru:
	"""docstring for error"""
	def __init__(self, Theta_2, D_theta2):
		self.Theta_2 = Theta_2
		self.D_theta2= D_theta2
i=0
Theta_2baru = []

for i in range(149):
 Theta_2baru.append(i)
 Theta_2baru[i]=Theta_2-(0.1*D_theta2[i])
 print("Theta_2_baru: %s" % (Theta_2baru[i]))
 i=i+1

class Theta_3_baru:
	"""docstring for error"""
	def __init__(self, Theta_3, D_theta3):
		self.Theta_3 = Theta_3
		self.D_theta3= D_theta3
i=0
Theta_3baru = []

for i in range(149):
 Theta_3baru.append(i)
 Theta_3baru[i]=Theta_3-(0.1*D_theta3[i])
 print("Theta_3_baru: %s" % (Theta_3baru[i]))
 i=i+1

class Theta_4_baru:
	"""docstring for error"""
	def __init__(self, Theta_4, D_theta4):
		self.Theta_4 = Theta_4
		self.D_theta4= D_theta4
i=0
Theta_4baru = []

for i in range(149):
 Theta_4baru.append(i)
 Theta_4baru[i]=Theta_4-(0.1*D_theta4[i])
 print("Theta_4_baru: %s" % (Theta_4baru[i]))
 i=i+1

class Theta_5_baru:
	"""docstring for error"""
	def __init__(self, Theta_5, D_theta5):
		self.Theta_5 = Theta_5
		self.D_theta5= D_theta5
i=0
Theta_5baru = []

for i in range(149):
 Theta_5baru.append(i)
 Theta_5baru[i]=Theta_5-(0.1*D_theta5[i])
 print("Theta_5_baru: %s" % (Theta_5baru[i]))
 i=i+1

class Theta_6_baru:
	"""docstring for error"""
	def __init__(self, Theta_6, D_theta6):
		self.Theta_6 = Theta_6
		self.D_theta6= D_theta6
i=0
Theta_6baru = []

for i in range(149):
 Theta_6baru.append(i)
 Theta_6baru[i]=Theta_6-(0.1*D_theta6[i])
 print("Theta_6_baru: %s" % (Theta_6baru[i]))
 i=i+1

class Theta_7_baru:
	"""docstring for error"""
	def __init__(self, Theta_7, D_theta7):
		self.Theta_7 = Theta_7
		self.D_theta7= D_theta7
i=0
Theta_7baru = []

for i in range(149):
 Theta_7baru.append(i)
 Theta_7baru[i]=Theta_7-(0.1*D_theta7[i])
 print("Theta_7_baru: %s" % (Theta_7baru[i]))
 i=i+1

class Theta_8_baru:
	"""docstring for error"""
	def __init__(self, Theta_8, D_theta8):
		self.Theta_8 = Theta_8
		self.D_theta8= D_theta8
i=0
Theta_8baru = []

for i in range(149):
 Theta_8baru.append(i)
 Theta_8baru[i]=Theta_8-(0.1*D_theta8[i])
 print("Theta_8_baru: %s" % (Theta_8baru[i]))
 i=i+1

class BiasBaru:
	"""docstring for error"""
	def __init__(self, Bias, D_bias):
		self.Bias = Bias
		self.D_bias= D_bias
i=0
BiasBaru = []

for i in range(149):
 BiasBaru.append(i)
 BiasBaru[i]=Bias-(0.1*D_bias[i])
 print("BiasBaru: %s" % (BiasBaru[i]))
 i=i+1

class BiasBaru:
	"""docstring for error"""
	def __init__(self, Bias, D_bias):
		self.Bias_2 = Bias_2
		self.D_bias2= D_bias2
i=0
BiasBaru2 = []

for i in range(149):
 BiasBaru2.append(i)
 BiasBaru2[i]=Bias_2-(0.1*D_bias2[i])
 print("BiasBaru2: %s" % (BiasBaru2[i]))
 i=i+1


print("Theta_1: %s" % (Theta_1))
print("Theta_2: %s" % (Theta_2))
print("Theta_3: %s" % (Theta_3))
print("Theta_4: %s" % (Theta_4))
print("Theta_5: %s" % (Theta_5))
print("Theta_6: %s" % (Theta_6))
print("Theta_7: %s" % (Theta_7))
print("Theta_8: %s" % (Theta_8))

# D_theta_1=
# D_theta_2=
# D_theta_3=
# D_theta_4=
# D_theta_5=
# D_theta_6=
# D_theta_7=
# D_theta_8=

# sigmoid=
# sigmoid_2=
# prediction=
# prediction_2=

