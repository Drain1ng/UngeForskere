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
def plotPaths(solutions,particle=2):
    colors = ["r","g","b"]
    if particle == 2:
        for i in range(len(solutions)):
            plt.scatter(solutions[i].x2, solutions[i].y2, s=1, color = colors[i])
        plt.title("Banekurver for arm 2")
    elif particle == 1:
        for i in range(len(solutions)):
            plt.scatter(solutions[i].x1, solutions[i].y1, s=1, color = colors[i])
        plt.title("Banekurver for arm 1")
    plt.xlabel("x /m")
    plt.ylabel("y /m")
    plt.show()
def sample(paramArray,nSamples):
    l = math.floor(len(paramArray)/nSamples)
    outArray = np.zeros(l+1)
    for i in range(l):
        outArray[i] = paramArray[i * nSamples]
    return outArray

def getTimeDiffs(solutions,j):
    dts = np.array([])
    for i in range(len(solutions)):
        k = len(solutions) - 1 - i
        while k != 0:
            dts = np.append(dts,solutions[i+1].time[j]-solutions[i].time[j])
            k -= 1
    return dts

def getAllTimeDiffs(solutions,shortestArrayLen, form = "abs"):
    l = shortestArrayLen
    out = np.zeros(l)
    for i in range(l):
        out[i] = np.max(getTimeDiffs(solutions,i))
    if form == "abs":
        return out
    elif form == "relative":
        return 100*out/(1/240)

def getMaxDistances(solutions,particle,shortestArrayLen):
    dmax = np.array([])
    for j in range(len(solutions[1].time[:shortestArrayLen])):
        dmax = np.append(dmax, np.max(getAllLengths(solutions,j,particle)))
    return dmax

def getAvgDistances(solutions,particle,shortestArrayLen):
    dmax = np.array([])
    for j in range(len(solutions[1].time[:shortestArrayLen])):
        dmax = np.append(dmax, np.average(getAllLengths(solutions,j,particle)))
    return dmax

def getDifference(array):
    out = np.zeros(len(array)-1)
    for i in range(len(out)):
        out[i] = array[i+1] - array[i]
    return out

def getAverage(array):
    out = np.zeros(len(array)-1)
    for i in range(len(out)):
        out[i] = (array[i+1] + array[i])/2
    return out

class Data:
    def __init__(self,path,nSamples=2):
        data = pd.read_excel (path)
        self.time = sample(pd.DataFrame(data, columns=['Tid']).to_numpy(),nSamples)
        self.theta1 = sample(pd.DataFrame(data, columns=['Vinkel1']).to_numpy(),nSamples)
        self.theta2 = sample(pd.DataFrame(data, columns=['Vinkel2']).to_numpy(),nSamples)
        self.x1 = np.sin(self.theta1) * l1
        self.y1 = -np.cos(self.theta1) * l1
        self.x2 = np.sin(self.theta2) * l2 + self.x1
        self.y2 = -np.cos(self.theta2) * l2 + self.y1
        self.omega1 = getDifference(self.theta1)/getDifference(self.time)
        self.omega2 = getDifference(self.theta2) / getDifference(self.time)
        self.omegaTime = getAverage(self.time)




d1 = Data(r'C:\Users\drain\CloudStation\3.D\UF\Data\Endeligt forsøg\4 ny.xlsx')
d2 = Data(r'C:\Users\drain\CloudStation\3.D\UF\Data\Endeligt forsøg\5.xlsx')
d3 = Data(r'C:\Users\drain\CloudStation\3.D\UF\Data\Endeligt forsøg\6.xlsx')

D=[d1,d2,d3]
shortestArrayLen = min(min(len(d1.time),len(d2.time)),len(d3.time))-1


plt.scatter(d2.time[:shortestArrayLen],getMaxDistances(D,1,shortestArrayLen)[:shortestArrayLen],s=3)
plt.title("Maksimal placeringsforskel mellem forsøg - arm 1",fontsize = 30)
plt.xlabel("t /s",fontsize = 20)
plt.ylabel("distance /m",fontsize = 20)
plt.show()


plt.scatter(D[0].omegaTime,D[0].omega1,s=0.5)
plt.scatter(D[0].omegaTime,D[0].omega2,s=0.5)
plt.xlabel("tid /s")
plt.ylabel("vinkelhastighed /rad/s")
plt.show()
getAllRelOmega(D,out="analysis",cutoff = 120*5)
print(d1.omegaTime[120*10])
getAllRelOmega(D,out="analysis",cutoff = 120*10)
print(d1.omegaTime[120*10])
getAllRelOmega(D,out="analysis",cutoff = 120*20)
print(d1.omegaTime[120*20])
getAllRelOmega(D,out="analysis",cutoff = 120*30)
print(d1.omegaTime[120*30])
getAllRelOmega(D,out="analysis")






#time diff
print("tidsforskelsanalyse:","(relative,abs) mean:", np.mean(getAllTimeDiffs(D,shortestArrayLen,form ="relative")),"  ",np.mean(getAllTimeDiffs(D,shortestArrayLen,form ="abs")),
"(relative,abs) std", np.std(getAllTimeDiffs(D,shortestArrayLen,form ="relative")),"  ", np.std(getAllTimeDiffs(D,shortestArrayLen,form ="abs")))

plt.scatter(range(shortestArrayLen),getAllTimeDiffs(D,shortestArrayLen,form ="abs")*10**3,s=1)
plt.xlabel("Billede nr.")
plt.ylabel("t /ms")
plt.show()
plt.scatter(range(shortestArrayLen),getAllTimeDiffs(D,shortestArrayLen,form ="relative"),s=1)
plt.xlabel("Billede nr.")
plt.ylabel("% af 1/240 s")
plt.show()



#placeringsforskel
plt.scatter(d2.time[:shortestArrayLen],getMaxDistances(D,1,shortestArrayLen)[:shortestArrayLen],s=3)
plt.title("maksimal placeringsforskel mellem forsøg - arm 1")
plt.xlabel("t /s")
plt.ylabel("distance /m")
plt.show()
plt.scatter(d2.time[:shortestArrayLen],getMaxDistances(D,2,shortestArrayLen)[:shortestArrayLen],s=3)
plt.title("maksimal placeringsforskel mellem forsøg - arm 2")
plt.xlabel("t /s")
plt.ylabel("distance /m")
plt.show()
plt.scatter(d2.time[:shortestArrayLen],getAvgDistances(D,1,shortestArrayLen)[:shortestArrayLen],s=3)
plt.title("Gennemsnitlig placeringsforskel mellem forsøg - arm 1")
plt.xlabel("t /s")
plt.ylabel("distance /m")
plt.show()
plt.scatter(d2.time[:shortestArrayLen],getAvgDistances(D,2,shortestArrayLen)[:shortestArrayLen],s=3)
plt.title("Gennemsnitlig placeringsforskel mellem forsøg - arm 2")
plt.xlabel("t /s")
plt.ylabel("distance /m")
plt.show()


#placeringforskelle kvantificeret:
f =getMaxDistances(D,2,shortestArrayLen)
print("Time when distance (+ =from left, - =from right, data = max curve)",
      "arm 2 max",
    "\n0.025 + m at", d1.time[getIndexWhereValueIs(0.025, f, 1)], "s",
    "\n0.025 - m at", d1.time[getIndexWhereValueIs(0.025, f, -1)], "s",
    "\n0.05 + m at",d1.time[getIndexWhereValueIs(0.05,f,1)],"s",
    "\n0.05 - m at",d1.time[getIndexWhereValueIs(0.05,f,-1)],"s",
    "\n0.1 + m at",d1.time[getIndexWhereValueIs(0.1,f,1)],"s",
    "\n0.1 - m at",d1.time[getIndexWhereValueIs(0.1,f,-1)],"s",
    "\n0.3 + m at", d1.time[getIndexWhereValueIs(0.3, f, 1)], "s",
    "\n0.3 - m at", d1.time[getIndexWhereValueIs(0.3, f, -1)], "s")
f =getMaxDistances(D,1,shortestArrayLen)
print("Time when distance (+ =from left, - =from right, data = max curve)",
      "arm 1 max",
    "\n0.025 + m at", d1.time[getIndexWhereValueIs(0.025, f, 1)], "s",
    "\n0.025 - m at", d1.time[getIndexWhereValueIs(0.025, f, -1)], "s",
    "\n0.05 + m at",d1.time[getIndexWhereValueIs(0.05,f,1)],"s",
    "\n0.05 - m at",d1.time[getIndexWhereValueIs(0.05,f,-1)],"s",
    "\n0.1 + m at",d1.time[getIndexWhereValueIs(0.1,f,1)],"s",
    "\n0.1 - m at",d1.time[getIndexWhereValueIs(0.1,f,-1)],"s",
    "\n0.3 + m at", d1.time[getIndexWhereValueIs(0.3, f, 1)], "s",
    "\n0.3 - m at", d1.time[getIndexWhereValueIs(0.3, f, -1)], "s")
f =getAvgDistances(D,2,shortestArrayLen)
print("Time when distance (+ =from left, - =from right, data = max curve)",
      "arm 2 avg",
    "\n0.025 + m at", d1.time[getIndexWhereValueIs(0.025, f, 1)], "s",
    "\n0.025 - m at", d1.time[getIndexWhereValueIs(0.025, f, -1)], "s",
    "\n0.05 + m at",d1.time[getIndexWhereValueIs(0.05,f,1)],"s",
    "\n0.05 - m at",d1.time[getIndexWhereValueIs(0.05,f,-1)],"s",
    "\n0.1 + m at",d1.time[getIndexWhereValueIs(0.1,f,1)],"s",
    "\n0.1 - m at",d1.time[getIndexWhereValueIs(0.1,f,-1)],"s",
    "\n0.3 + m at", d1.time[getIndexWhereValueIs(0.3, f, 1)], "s",
    "\n0.3 - m at", d1.time[getIndexWhereValueIs(0.3, f, -1)], "s")
f =getAvgDistances(D,1,shortestArrayLen)
print("Time when distance (+ =from left, - =from right, data = max curve)",
      "arm 1 avg",
    "\n0.025 + m at", d1.time[getIndexWhereValueIs(0.025, f, 1)], "s",
    "\n0.025 - m at", d1.time[getIndexWhereValueIs(0.025, f, -1)], "s",
    "\n0.05 + m at",d1.time[getIndexWhereValueIs(0.05,f,1)],"s",
    "\n0.05 - m at",d1.time[getIndexWhereValueIs(0.05,f,-1)],"s",
    "\n0.1 + m at",d1.time[getIndexWhereValueIs(0.1,f,1)],"s",
    "\n0.1 - m at",d1.time[getIndexWhereValueIs(0.1,f,-1)],"s",
    "\n0.3 + m at", d1.time[getIndexWhereValueIs(0.3, f, 1)], "s",
    "\n0.3 - m at", d1.time[getIndexWhereValueIs(0.3, f, -1)], "s")



#paths
plotPaths(D,particle=2)
plotPaths(D,particle=1)

#vinkler
plotTheta(D,particle=2)
plotTheta(D,particle=1)