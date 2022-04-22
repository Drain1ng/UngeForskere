import numpy as np
import math
import matplotlib.pyplot as plt

def getDifferenceVectorLength(x1,y1,x2,y2):
    dx =x2-x1
    dy =y2-y1
    length = math.sqrt(dx**2+dy**2)
    return length

def getAllLengths(solutions,j,particle):
    lengths = np.array([])
    if particle == 2:
        for i in range(len(solutions)):
            k = len(solutions) - 1 - i
            while k != 0:
                v = getDifferenceVectorLength(solutions[i].x2[j], solutions[i].y2[j], solutions[i + 1].x2[j], solutions[i + 1].y2[j])
                lengths = np.append(lengths, v)
                k -= 1
    elif particle == 1:
        for i in range(len(solutions)):
            k = len(solutions) - 1 - i
            while k != 0:
                v = getDifferenceVectorLength(solutions[i].x1[j], solutions[i].y1[j], solutions[i + 1].x1[j],
                                              solutions[i + 1].y1[j])
                lengths = np.append(lengths, v)
                k -= 1
    return lengths



def getMaxDistances(solutions,particle):
    dmax = np.array([])
    for j in range(len(solutions[1].time)):
        dmax = np.append(dmax, np.max(getAllLengths(solutions,j,particle)))
    return dmax

def getAvgDistances(solutions,particle):
    davg = np.array([])
    for j in range(len(solutions[1].time)):
        davg = np.append(davg, np.mean(getAllLengths(solutions,j,particle)))
    return davg

def getMinDistances(solutions,particle):
    dmin = np.array([])
    for j in range(len(solutions[1].time)):
        dmin = np.append(dmin, np.min(getAllLengths(solutions,j,particle)))
    return dmin



def getReferenceDistance(solutions, referenceIndex,j,particle):
    referenceSolution = solutions[referenceIndex]
    referenceDistance = np.array([])
    if particle == 2:
        for i in range(len(solutions)-1):
            d = getDifferenceVectorLength(referenceSolution.x2[j],referenceSolution.y2[j],solutions[i+1].x2[j],solutions[i+1].y2[j])
            referenceDistance = np.append(referenceDistance,d)
    elif particle == 1:
        for i in range(len(solutions) - 1):
            d = getDifferenceVectorLength(referenceSolution.x1[j], referenceSolution.y1[j], solutions[i + 1].x1[j],
                                          solutions[i + 1].y1[j])
            referenceDistance = np.append(referenceDistance, d)
    return referenceDistance

def getAvgReferenceDistances(solutions, referenceIndex,particle):
    ard = np.array([])
    for j in range(len(solutions[0].time)):
        d = getReferenceDistance(solutions, referenceIndex,j,particle)
        ard = np.append(ard,np.average(d))
    return ard

def getMaxReferenceDistances(solutions, referenceIndex,particle):
    mrd = np.array([])
    for j in range(len(solutions[0].time)):
        d = getReferenceDistance(solutions, referenceIndex,j,particle)
        mrd = np.append(mrd,np.max(d))
    return mrd

def getMinReferenceDistances(solutions, referenceIndex,particle):
    mrd = np.array([])
    for j in range(len(solutions[0].time)):
        d = getReferenceDistance(solutions, referenceIndex,j,particle)
        mrd = np.append(mrd,np.min(d))
    return mrd

def getIndexWhereValueIs(cutoff,distances,dir):
    #print("getIndexWhereValueIs")
    if dir == 1:
        for i in range(len(distances)):
            #print(distances[i], ">", cutoff)
            if distances[i] > cutoff:
                #print("getIndexWhereValueIs","i:",i," dist:", distances[i])
                return i
    if dir == -1:
        for i in range(len(distances)):
            #print(distances[-i], ">", cutoff)
            if distances[-i] > cutoff:
                #print("getIndexWhereValueIs","i:",len(distances)-i," dist:", distances[-i])
                return len(distances)-i
    return 0


def getAvgE(solutions,particle=0):
    a = np.zeros(solutions[0].Emek.shape)
    if particle == 0:
        for i in solutions:
            a += i.Emek
        a = a/len(solutions)
    elif particle == 1:
        for i in solutions:
            a += i.Emek1
        a = a / len(solutions)
    elif particle == 2:
        for i in solutions:
            a += i.Emek2
        a = a / len(solutions)
    return a

def getAvgEpot(solutions,particle=0):
    a = np.zeros(solutions[0].Emek.shape)
    if particle == 0:
        for i in solutions:
            a += i.Epot
        a = a/len(solutions)
    if particle == 1:
        for i in solutions:
            a += i.Epot1
        a = a/len(solutions)
    if particle == 2:
        for i in solutions:
            a += i.Epot2
        a = a/len(solutions)
    return a

def getAvgEkin(solutions,particle=0):
    a = np.zeros(solutions[0].Emek.shape)
    if particle == 0:
        for i in solutions:
            a += i.Ekin
        a = a/len(solutions)
    if particle == 1:
        for i in solutions:
            a += i.Ekin1
        a = a/len(solutions)
    if particle == 2:
        for i in solutions:
            a += i.Ekin2
        a = a/len(solutions)
    return a



def getETypeDisttribution(solutions):
    E = getAvgE(solutions)
    Epot = (getAvgEpot(solutions))/E
    Ekin = getAvgEkin(solutions) / E
    return np.array([Epot,Ekin])

def plotETypeDisttribution(solutions):
    plt.scatter(solutions[0].time, getETypeDisttribution(solutions)[1], s=0.1)
    plt.scatter(solutions[0].time, getETypeDisttribution(solutions)[0], s=0.1)
    plt.title("Fordeling af energityper")
    plt.xlabel("t /s")
    plt.ylabel("Andel af den totale energi")
    plt.show()

def getEParticleDistribrution(solutions):
    E = getAvgE(solutions)
    E1 = (getAvgE(solutions, particle=1))/E
    E2 = (getAvgE(solutions, particle=2))/E
    return np.array([E1,E2])

def plotEParticleDistribrution(solutions):
    plt.scatter(solutions[0].time, getEParticleDistribrution(solutions)[1], s=0.1)
    plt.scatter(solutions[0].time, getEParticleDistribrution(solutions)[0], s=0.1)
    plt.title("Fordeling af energi i armene")
    plt.xlabel("t /s")
    plt.ylabel("Andel af den totale energi")
    plt.show()

def getEParticle(solutions):
    E1 = (getAvgE(solutions, particle=1))
    E2 = (getAvgE(solutions, particle=2))
    return np.array([E1,E2])

def plotEParticle(solutions):
    plt.scatter(solutions[0].time, getEParticle(solutions)[1], s=0.1)
    plt.scatter(solutions[0].time, getEParticle(solutions)[0], s=0.1)
    plt.title("Energi i hver arm")
    plt.xlabel("t /s")
    plt.ylabel("E /J")
    plt.show()


def getRelativeOmega(solution,cuttof=0):
    if cuttof == 0:
        ar = np.zeros(np.shape(solution.omega1))
    else:
        ar = np.zeros(np.shape(solution.omega1[:cuttof]))
    for i in range(len(ar)):
        if solution.omega1[i] * solution.omega2[i] < 0:
            ar[i] += 1
        elif (solution.omega1[i] == 0 or solution.omega2[i] == 0):
            ar[i] += 0.5
            print("THIS IS BS")
        else:
            ar[i] += 0
    return ar

def getAvgRelOmega(solutions,out="array"):
    for i in range(len(solutions)):
        if i == 0:
            ar = getRelativeOmega(solutions[i])
        else:
            ar += getRelativeOmega(solutions[i])
    ar = ar/len(solutions)
    if out == "array":
        return ar
    elif out == "number":
        return np.mean(ar)

def getAllRelOmega(solutions,out="standard",cutoff=0):
    all = np.zeros(0)
    opposite = np.zeros(0)
    same = np.zeros(0)
    r = range(len(solutions))
    for i in r:
        k=np.mean(getRelativeOmega(solutions[i],cutoff))
        all = np.append(all,k)
        if k > 0.5:
            opposite = np.append(opposite,k)
        elif k < 0.5:
            same = np.append(same,k)
    if out == "standard":
        return all
    if out == "analysis":
        return print("\nrelative omega analysis:\n",
                     "mean: ", np.mean(all),
                     "\nall solutions:\n", all,
                     "\n opposite solutions mean: ", np.mean(opposite),
                     "\n same solutions mean: ", np.mean(same),
                     "\n cutoff: ", cutoff)


def addThetaPlot(solution,particle=2):
    if particle == 1:
        plt.scatter(solution.time, solution.theta1, s=0.1)
    elif particle == 2:
        plt.scatter(solution.time, solution.theta2, s=0.1)


def plotTheta(solutions,particle=2):
    for i in range(len(solutions)):
        addThetaPlot(solutions[i], particle=particle)
    if particle == 2:
        plt.title("vinkel 2")
    else:
        plt.title("vinkel 1")
    plt.xlabel("t /s")
    plt.ylabel("vinkel /rad")
    plt.show()

def getK(solution):
    wf = np.zeros((np.shape(solution.Emek1)[0] - 1, 1))
    r = range(len(wf))
    for i in r:
        wf[i] = -(solution.Emek[i + 1] - solution.Emek[i]) / 2
        if wf[i] < 0:
            print("FEJL", i)
    sv = np.zeros(np.shape(wf))
    for i in r:
        b = (solution.v1[i] ** 2) + (solution.v1[i] ** 2)
        c = (solution.v1[i + 1] ** 2) + (solution.v1[i + 1] ** 2)
        sv[i] = (b + c) / 2
    k = wf / sv
    return k

def getRotations(solution,output="number",particle=2, correction =0):
    rotations = np.zeros(np.shape(solution.theta2))
    n = 0
    dn = 0.1
    if particle == 2:
        for i in range(len(solution.theta2)-1):
            if np.abs(solution.theta2[i+1] - solution.theta2[i]) > np.pi:
                n += dn
            rotations[i] = n
    if particle == 1:
        for i in range(len(solution.theta1)-1):
            if np.abs(solution.theta1[i+1] - solution.theta1[i]) > np.pi:
                n += dn
            rotations[i] = n
    if output == "number":
        return round((np.max(n)/dn) - correction,0)
    if output == "plot":
        addThetaPlot(solution, particle=particle)
        plt.scatter(solution.time,rotations,s=0.5)
        plt.title("scatter")
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.show()

def plotDistances(solutions,particle):
    plt.scatter(solutions[0].time, getMinDistances(solutions, particle), s=1, label="Min")
    plt.scatter(solutions[0].time, getMaxDistances(solutions, particle), s=1, label="Max")
    plt.scatter(solutions[0].time, getAvgDistances(solutions, particle), s=1, label="Avg")
    plt.title("All distances between solutions")
    plt.xlabel("t /s")
    plt.ylabel("distance /m")
    plt.legend(loc='upper center', markerscale=5)
    plt.show()