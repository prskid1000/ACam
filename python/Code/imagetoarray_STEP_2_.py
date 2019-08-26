import numpy as np
import matplotlib.pyplot as plt
import os
y,z=[],[]
p=""
p=input("Enter the path where you extracted aimage folder ending with / = ")
dat=["eraser","pen","sharpner","pencil"]

for d in dat:
   dirs=os.listdir(p+'aimage/'+d)
   i=0
   for file in dirs:
       x=plt.imread(p+'aimage/'+d+"/"+file)
       y.append(x)
       z.append(d)
       i=i+1
   print(len(y))
   print(y[0].shape)
y=np.array(y)
z=np.array(z)
np.savez_compressed(p+"/aimage/data",x=y,y=z)
print(len(z))
print(len(y))
