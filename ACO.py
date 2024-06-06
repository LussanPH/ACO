import numpy as np
import random as rd
from sklearn.model_selection import train_test_split
class Formiga:
    def __init__(self):
        self.alpha = rd.randint(1,3)
        self.beta = rd.randint(1,3)
        self.caminho = []

class ACO:
    def __init__(self, num, model):
        self.num = num
        self.model = model

    def matrizAdj(self):
        if(self.model == 0):
            self.matadj = np.zeros((16, 16), dtype=np.float64)
            self.matadj[0, 1], self.matadj[1, 2], self.matadj[1, 3], self.matadj[1, 4], self.matadj[1,5], self.matadj[2, 2], self.matadj[2, 3], self.matadj[2, 4], self.matadj[2, 5], self.matadj[2, 6], self.matadj[3,3], self.matadj[3,4], self.matadj[3,5], self.matadj[3,6], self.matadj[3, 2], self.matadj[4, 2], self.matadj[4, 3], self.matadj[4,4], self.matadj[4,5], self.matadj[4,6], self.matadj[5,2], self.matadj[5,3], self.matadj[5,4], self.matadj[5,5], self.matadj[5,6], self.matadj[6,7], self.matadj[6,8], self.matadj[6,9], self.matadj[7,10], self.matadj[8,10], self.matadj[9,10], self.matadj[10,11], self.matadj[10,12], self.matadj[10,13], self.matadj[10,14], self.matadj[11,11], self.matadj[11,12], self.matadj[11,13], self.matadj[11,14], self.matadj[11,15], self.matadj[12,11], self.matadj[12,12], self.matadj[12,13], self.matadj[12,14], self.matadj[12,15], self.matadj[13,11], self.matadj[13,12], self.matadj[13,13], self.matadj[13,14], self.matadj[13,15], self.matadj[14,11], self.matadj[14,12], self.matadj[14,13], self.matadj[14,14], self.matadj[14,15] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            self.matrizPos()
            self.matrizFer()

    def matrizPos(self):
        self.matPos = np.zeros((16, 16), dtype=np.float64)
        if(self.model == 0):
            for i in range(len(self.matadj)):
                for j in range(len(self.matadj)):
                    if(self.matadj[i,j] == 1):
                        self.matPos[i, j] = 0.5                       

    def matrizFer(self):
        self.matFer = np.zeros((16, 16), dtype=np.float64)
        if(self.model == 0):
            for i in range(len(self.matadj)):
                for j in range(len(self.matadj)):
                    if(self.matadj[i,j] == 1):
                        self.matFer[i, j] = 5 

    def gerarFormiga(self):
        f = Formiga()
        return f
    
    def calculoPorcentagem(self, fer, pos, f, c):#fer e pos são listas de ferormonios e distancias do vertice
        lista = []
        indices = []
        for i in range(c):
            soma += (pos[c]**f.alpha) * (fer[c]**f.beta)
            lista.append(0)
            indices.append(i)
        for i in range(len(lista)):
            p = ((pos[i]**f.alpha) * (fer[i]**f.beta))/2
            lista[i] = p
        escolhido = np.random.choice(indices, p=lista)
        return escolhido

        

aco = ACO(1, 0)
aco.matrizAdj()