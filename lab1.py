import math
from random import randint, random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import numpy as np
import sys
import random


class Vertex:  # ok
    def __init__(self, index, x, y, value):
        self.index = index
        self.x = x
        self.y = y
        self.value = value
        self.isUsed = False
        # tablica kosztow przejsc z kazdym
        self.next = self
        self.previous = self

    def setUsed(self):
        self.isUsed = True

    def setUnused(self):
        self.isUsed = False

    def setNext(self, vertex):
        self.next = vertex

    def setPrevious(self, vertex):
        self.previous = vertex

class Path:
    def __init__(self, Vertex):
        self.vertexArray = []
        self.gain = 0.0
        self.addFirstVertex(Vertex)
        self.value = 0.0

    def getLastVertex(self):
        return self.vertexArray[len(self.vertexArray) - 1]

    def addVertex(self, Vertex):
        self.gain += float(computeGain(self.getLastVertex(), Vertex))
        self.vertexArray.append(Vertex)
        Vertex.setUsed()

    def addCycleVertex(self, element, Vertex):
        self.vertexArray.append(Vertex)
        Vertex.setUsed()
        Vertex.setPrevious(element)
        Vertex.setNext(element.next)
        element.next.setPrevious(Vertex)
        element.setNext(Vertex)

    def addSecondCycleElement(self, element, Vertex):
        self.vertexArray.append(Vertex)
        Vertex.setUsed()
        Vertex.setPrevious(element)
        Vertex.setNext(element)
        element.setNext(Vertex)
        element.setPrevious(Vertex)

    def addThirdCycleElement(self, element, Vertex):
        self.vertexArray.append(Vertex)
        Vertex.setUsed()
        element.next.setNext(Vertex)
        Vertex.setNext(element)
        element.setPrevious(Vertex)
        Vertex.setPrevious(element.next)

    def computeCycleValue(self):
        gain = 0
        for index in range(0, len(self.vertexArray)):
            gain += self.vertexArray[index].value
            gain = gain - computeCost(self.vertexArray[index], self.vertexArray[index].next)
        self.value = gain

    def addLastVertex(self, Vertex):
        self.gain -= float(computeCost(self.getLastVertex(), Vertex))
        self.vertexArray.append(Vertex)

    def addFirstVertex(self, Vertex):
        self.gain += Vertex.value
        self.vertexArray.append(Vertex)
        Vertex.setUsed()

    def sortCycleArray(self):
        sortedArray = [0] * (len(self.vertexArray) + 1)
        sortedArray[0] = self.vertexArray[0]
        for i in range(1, len(self.vertexArray)):
            sortedArray[i] = sortedArray[i - 1].next

        sortedArray[len(self.vertexArray)] = self.vertexArray[0]
        self.vertexArray = sortedArray

    def printPath(self, t):
        array = []
        for vertex in self.vertexArray:
            array.append(resolvePointString(vertex, t))
        return array

    def printCoordinatesX(self):
        array = []
        for vertex in self.vertexArray:
            array.append(vertex.x)
        return array

    def printCoordinatesY(self):
        array = []
        for vertex in self.vertexArray:
            array.append(vertex.y)
        return array

    def plotPath(self):
        x, y = zip(*self.vertexArray)
        plt.scatter(x, y)

    def getPathLength(self):
        return len(self.vertexArray)

    def computePathValue(self):
        gain = 0
        cost = 0
        for point in self.vertexArray:
            gain += point.value

        for connection in self.connections:
            cost += connection.computeCost()

        self.value = (gain - cost)

    def computePathValueNN(self):
        gain = 0
        cost = 0
        for index in range(0, len(self.vertexArray) - 1):
            gain += self.vertexArray[index].value
            cost += computeCost(self.vertexArray[index], self.vertexArray[index + 1])

        self.value = (gain - cost)

def plotPoints(pointsArray):
    x, y = zip(*zip(pointsArray, y))
    x, y = zip(pointsArray)
    plt.scatter(x, y)

def computeCost(start, end):
    return (math.sqrt(math.pow((end.x - start.x), 2) + math.pow((end.y - start.y), 2)) * 5)


def computeGain(start, end):
    return end.value - computeCost(start, end)

def resolvePointString(point, t):
    if t == 'c':
        return "[" + str(point.x) + ", " + str(point.y) + "]"
    if t == 'i':
        return str(point.index)

def resetPoints(pointsArray):
    for point in pointsArray:
        point.setUnused()
        point.next = point
        point.previous = point

def resolvePointArray(point):
    return np.array(point.x, point.y)

def computeGainCycle(start, end):
    value = end.value
    costNow = computeCost(start, start.next)
    costAfter = computeCost(start, end) + computeCost(end, start.next)
    gain = value + costNow - costAfter
    return gain

def readData():
    points = pd.read_csv(open('./data/kroa100.csv'), delimiter=' ')
    gains = pd.read_csv(open('./data/krob100.csv'), delimiter=' ')

    del gains['i']
    del gains['y']

    pointsArray = []
    pathsArray = []
    gains = gains.values

    for point in points.values:
        vertex = Vertex(point[0] - 1, point[1], point[2], gains[point[0] - 1])
        pointsArray.append(vertex)

    costMatrix = np.ndarray((len(pointsArray), len(pointsArray)))
    for vertex in pointsArray:
        for vertex2 in pointsArray:
            costMatrix[vertex.index, vertex2.index] = computeCost(vertex, vertex2)

    return [pointsArray, pathsArray, gains, costMatrix]

def RandomPath():
    [pointsArray, pathsArray, gains, costMatrix] = readData()
    maxGain = -sys.float_info.max
    minGain = sys.float_info.max
    suma = 0
    f2 = open('randomPath.txt', 'w')

    for root in pointsArray:
        max = -1
        resetPoints(pointsArray)
        choicePointArray = pointsArray[:]
        path = Path(root)
        choicePointArray.remove(root)
        pointNo = randint(3, 99)
        for i in range(0, pointNo-1):
            toAdd = random.choice(choicePointArray)
            path.addVertex(toAdd)
            choicePointArray.remove(toAdd)

        path.addLastVertex(root)
        f2.write(' '.join(path.printPath('i')) + '\n')

    #     if path.gain > maxGain:
    #         maxGain = path.gain
    #         bestPath = path
    #         # break
    #     if path.gain < minGain:
    #         minGain = path.gain
    #         worstPath = path
    #     suma += path.gain
    # meanPath = suma / len(pointsArray)

    # f2.write(' '.join(path.printPath('i')) + '\n')

def GreedyNN():
    [pointsArray, pathsArray, gains, costMatrix] = readData()
    maxGain = 0
    minGain = sys.float_info.max
    maxPathValue = 0
    suma = 0
    # f = open('greedyNN.txt', 'w')
    f = open('greedyNNPath.txt', 'w')
    # f2 = open('greedyNNPath.txt', 'w')

    for root in pointsArray:  # dla wsyzstkich mozliwych punktow startowych
        max = -1
        resetPoints(pointsArray)
        path = Path(root)
        while True:  # wykonuje petle
            for vertex2add in pointsArray:  # po wszystkich punktach
                if not vertex2add.isUsed:  # ktore jeszcze nie sa wykorzystane
                    addedGain = computeGain(path.getLastVertex(), vertex2add)
                    if addedGain > max:
                        toAdd = vertex2add
                        max = addedGain

            if max >= 0:  # majac wyznaczony najlepszy punkt
                path.addVertex(toAdd)  # dodaje do sciezki punkt
                max = -1
            else:
                # suma += path.gain
                break

        path.addLastVertex(root)  # dodaje zamkniecie

        # f.write("GreedyNN" + " profit " + str(path.gain) + " vertexes:" + ' '.join(path.printPath('i')) + '\n')
        f.write(' '.join(path.printPath('i')) + '\n')
        if path.gain > maxGain:
            maxGain = path.gain
            bestPath = path
            # break
        if path.gain < minGain:
            minGain = path.gain
            worstPath = path
        suma += path.gain
    meanPath = suma / len(pointsArray)

    # f.write("BEST: " + " profit " + str(bestPath.gain) + " vertexes:" + ' '.join(bestPath.printPath('i')) + '\n')
    # f.write("BEST: " + " profit " + str(bestPath.gain) + " coordinates:" + ' '.join(bestPath.printPath('c')) + '\n')
    # f.write("WORST: " + " profit " + str(worstPath.gain) + " vertexes:" + ' '.join(worstPath.printPath('i')) + '\n')
    # f.write("WORST: " + " profit " + str(suma) + " coordinates:" + ' '.join(worstPath.printPath('c')) + '\n')
    # f.write("MEAN: " + " profit " + str(meanPath) + '\n')
    # f.write(str(bestPath.printCoordinatesX()) + '\n')
    # f.write(str(bestPath.printCoordinatesY()) + '\n')
    # f2.write(' '.join(bestPath.printPath('i')))
    f.close()
    print(bestPath.printPath('i'))
    print(bestPath.gain)
    print(len(bestPath.vertexArray))
    print(len(pointsArray))

def GreedyCycle():
    [pointsArray, pathsArray, gains, costMatrix] = readData()
    maxGain = 0
    minGain = sys.float_info.max
    maxPathValue = 0
    suma = 0
    f = open('greedyCyclePath.txt', 'w')
    # f2 = open('greedyCyclePath.txt', 'w')

    for root in pointsArray:  # dla wsyzstkich mozliwych punktow startowych
        max = -1
        resetPoints(pointsArray)
        path = Path(root)
        # print(str(root) + str(root.x))

        for vertex2add in pointsArray:  # po wszystkich punktach
            if not vertex2add.isUsed:  # ktore jeszcze nie sa wykorzystane
                addedGain = computeGain(root, vertex2add)
                if addedGain > max:
                    toAdd = vertex2add
                    max = addedGain

        if max >= 0:  # majac wyznaczony najlepszy punkt
            path.addCycleVertex(root, toAdd)  # dodaje do sciezki punkt
            max = -1
        else:
            # print(path.printPath('i'))

            path.computeCycleValue()
            if path.value > maxPathValue:
                maxPathValue = path.value
                bestPath = path
            if path.value < minGain:
                minGain = path.value
                worstPath = path
            suma += path.value
            path.sortCycleArray()
            # f.write("GreedyCycle" + " profit " + str(path.value) + " vertexes:" + ' '.join(path.printPath('i')) + '\n')
            f.write(' '.join(path.printPath('i')) + '\n')
            continue

        for vertex2add in pointsArray:  # po wszystkich punktach
            if not vertex2add.isUsed:  # ktore jeszcze nie sa wykorzystane
                addedGain = computeGainCycle(path.vertexArray[1], vertex2add)
                if addedGain > max:
                    toAdd = vertex2add
                    max = addedGain

        if max >= 0:  # majac wyznaczony najlepszy punkt
            path.addCycleVertex(path.vertexArray[1], toAdd)  # dodaje do sciezki punkt
            max = -1
        else:
            print(path.printPath('i'))
            path.computeCycleValue()
            if path.value > maxPathValue:
                maxPathValue = path.value
                bestPath = path
            if path.value < minGain:
                minGain = path.value
                worstPath = path
            suma += path.gain
            path.sortCycleArray()
            # f.write("GreedyCycle" + " profit " + str(path.value) + " vertexes:" + ' '.join(path.printPath('i')) + '\n')
            f.write(' '.join(path.printPath('i')) + '\n')
            continue

        while True:  # wykonuje petle
            for pathElement in path.vertexArray:  # po wszystkich punktach w sciezce
                for vertex2add in pointsArray:  # po wszystkich punktach
                    if not vertex2add.isUsed:  # ktore jeszcze nie sa wykorzystane
                        addedGain = computeGainCycle(pathElement, vertex2add)
                        if addedGain > max:
                            toAdd = vertex2add
                            max = addedGain
                            element = pathElement

            if max >= 0:  # majac wyznaczony najlepszy punkt
                path.addCycleVertex(element, toAdd)  # dodaje do sciezki punkt
                max = -1
            else:
                break

        path.computeCycleValue()
        if path.value > maxPathValue:
            maxPathValue = path.value
            bestPath = path
        if path.value < minGain:
            minGain = path.value
            worstPath = path
        suma += path.value
        path.sortCycleArray()
        # f.write("GreedyCycle" + " profit " + str(path.value) + " vertexes:" + ' '.join(path.printPath('i')) + '\n')
        f.write(' '.join(path.printPath('i')) + '\n')
    meanPath = suma / len(pointsArray)
    # f.write("BEST: " + " profit " + str(bestPath.value) + " vertexes:" + ' '.join(bestPath.printPath('i')) + '\n')
    # f.write("BEST: " + " profit " + str(bestPath.value) + " coordinates:" + ' '.join(bestPath.printPath('c')) + '\n')
    # f.write("WORST: " + " profit " + str(worstPath.value) + " vertexes:" + ' '.join(worstPath.printPath('i')) + '\n')
    # f.write("MEAN: " + " profit " + str(meanPath) + '\n')
    # f.write(str(bestPath.printCoordinatesX()) + '\n')
    # f.write(str(bestPath.printCoordinatesY()) + '\n')
    # f.write(' '.join(bestPath.printPath('c')) + '\n')
    # f.close()
    # f2.write(' '.join(bestPath.printPath('i')))
    # f2.close()
    f.close()
    # path.sortCycleArray()
    # print(path.printPath('i'))


    print(bestPath.printPath('i'))
    print(bestPath.value)
    print(len(bestPath.vertexArray))

# GreedyNN()
GreedyCycle()
# RandomPath()
