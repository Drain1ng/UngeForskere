import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from Functions import *


l1=19/100
l2=13/100
g=9.82
m1 = 27.14/1000
m2 = 26.335/1000



class Solution:
    def __init__(self,path):
        data = pd.read_excel (path)
        self.time=pd.DataFrame(data, columns= ['tid']).to_numpy()
        self.theta1 = pd.DataFrame(data, columns= ['vinkel1k']).to_numpy()
        self.x1= np.sin(self.theta1)*l1
        self.y1 = -np.cos(self.theta1) * l1
        self.theta2 = pd.DataFrame(data, columns= ['vinkel2k']).to_numpy()
        self.x2 = np.sin(self.theta2) * l2 +self.x1
        self.y2 = -np.cos(self.theta2) * l2 + self.y1
        self.omega1 = pd.DataFrame(data, columns= ['vinkelhastighed1']).to_numpy()
        self.v1 = self.omega1*l1
        self.Epot1 = m1 * g * (self.y1+l1)
        self.Ekin1 = 1/2 * m1 * (self.v1**2)
        self.Emek1 = self.Epot1 + self.Ekin1

        self.omega2 = pd.DataFrame(data, columns=['vinkelhastighed2']).to_numpy()
        self.omega1x = np.cos(self.theta1) * self.omega1 * l1
        self.omega1y = np.sin(self.theta1) * self.omega1 * l1
        self.omega2x = np.cos(self.theta2) * self.omega2 *l2 + self.omega1x
        self.omega2y = np.sin(self.theta2) * self.omega2 *l2 + self.omega1y
        self.v2 = np.sqrt(np.power(self.omega2x,2)+np.power(self.omega2y,2))
        self.Epot2 = m2 * g * (self.y2+l1+l2)
        self.Ekin2 = 1/2 * m2 * (self.v2**2)
        self.Emek2 = self.Epot2 + self.Ekin2
        self.Ekin = self.Ekin1 + self.Ekin2
        self.Epot = self.Epot1 + self.Epot2
        self.Emek = self.Emek1 + self.Emek2




s1 = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\t0.xlsx')
s2 =Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\t1.xlsx')
s3 =Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\t-1.xlsx')
s4  = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\t2.xlsx')
s5 = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\t-2.xlsx')
s6 = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\t3.xlsx')
s7 = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\t-3.xlsx')
s8 = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\t4.xlsx')
s9 = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\t-4.xlsx')
s10 = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\t5.xlsx')
s11 = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\t-5.xlsx')
S = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11]


o1 = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\o0.xlsx')
o2 = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\05.xlsx')
o3 = Solution(r'C:\Users\drain\CloudStation\3.D\UF\numerisk løsning excelarray\o-5.xlsx')

O=[o1,o2,o3]
#plt.scatter(s1.time,S[0].Epot,s=0.5, label = "Epot")
#plt.scatter(s1.time,S[0].Ekin,s=0.5, label = "Ekin")
#plt.scatter(s1.time,S[0].Emek,s=2, label = "Emek")
#plt.scatter(s1.time,S[0].v1,s=0.5)
#plt.scatter(s1.time,S[0].v2,s=0.5)
#plotEParticle(S)
#plotEParticleDistribrution(S)
#plotETypeDisttribution(S)
#plt.scatter(s1.time,getAvgEpot(S),s=0.1)
#plt.scatter(s1.time,getAvgEkin(S),s=0.1)
#plt.scatter(s1.time,getAvgE(S),s=1)
#plt.legend(loc='upper center', markerscale=5)
#plt.title("Energy in system")
#plt.xlabel("t /s")
#plt.ylabel("E /J")
#plt.show()

plt.scatter(s1.time,getMaxDistances(S,1),s=3)
plt.title("Maksimal placeringsforskel mellem forsøg - arm 1",fontsize = 30)
plt.xlabel("t /s",fontsize = 20)
plt.ylabel("distance /m",fontsize = 20)
plt.show()

#plt.scatter(s1.time,getMaxReferenceDistances(S,0,2),s=1)
#plt.scatter(s1.time,getMinReferenceDistances(S,0,2),s=1)
#plt.scatter(s1.time,getAvgReferenceDistances(S,0),s=1)

plt.scatter(s1.x2,s1.y2,s=0.5)
plt.scatter(s2.x2,s2.y2,s=0.5)
plt.scatter(s3.x2,s3.y2,s=0.5)
plt.title("Banekurver for arm 2")
plt.xlabel("x /m")
plt.ylabel("y /m")
plt.show()

#plt.scatter(s1.time,getRelativeOmega(s3),s=0.5)



#plotDistances(S,1)

#for i in range(len(S)):
   #getRotations(S[i], output="plot")
  # getRotations(S[i])

#print(getAvgRelOmega(S, out="number"))

#print("rotations:",
#      "\n",getRotations(S[0],correction =6),
#      "\n",getRotations(S[1],correction =2),
#      "\n",getRotations(S[2],correction =4),
#      "\n",getRotations(S[3],correction =4),
#      "\n",getRotations(S[4],correction =2),
#     "\n",getRotations(S[5],correction =2),
#      "\n",getRotations(S[6],correction =2),
#      "\n",getRotations(S[7],correction =2),
#      "\n",getRotations(S[8],correction =2),
#      "\n",getRotations(S[9],correction =2),
#      "\n",getRotations(S[10],correction =6),
#      "\nmean: ", np.mean(np.array([getRotations(S[0],correction =6),getRotations(S[1],correction =2),getRotations(S[2],correction =4),getRotations(S[3],correction =4),getRotations(S[4],correction =2),getRotations(S[5],correction =2),getRotations(S[6],correction =2),getRotations(S[7],correction =2),getRotations(S[8],correction =2),getRotations(S[9],correction =2),getRotations(S[10],correction =6)])))

#plotTheta(S,particle=2)
#plotTheta(S,particle=1)
getAllRelOmega(O,out="analysis")

#placeringforskelle kvantificeret:
f =getMaxDistances(S,2)
print("Time when distance (+ =from left, - =from right, data = max curve)",
      "arm 2 max",
    "\n0.025 + m at", o1.time[getIndexWhereValueIs(0.025, f, 1)], "s",
    "\n0.025 - m at", o1.time[getIndexWhereValueIs(0.025, f, -1)], "s",
    "\n0.05 + m at",o1.time[getIndexWhereValueIs(0.05,f,1)],"s",
    "\n0.05 - m at",o1.time[getIndexWhereValueIs(0.05,f,-1)],"s",
    "\n0.1 + m at",o1.time[getIndexWhereValueIs(0.1,f,1)],"s",
    "\n0.1 - m at",o1.time[getIndexWhereValueIs(0.1,f,-1)],"s",
    "\n0.3 + m at", o1.time[getIndexWhereValueIs(0.3, f, 1)], "s",
    "\n0.3 - m at", o1.time[getIndexWhereValueIs(0.3, f, -1)], "s")