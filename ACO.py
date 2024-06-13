import numpy as np
import random as rd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from math import e
from math import log


class Formiga:
    def __init__(self):
        self.alpha = rd.randint(1,3)
        self.beta = rd.randint(1,3)
        self.caminho = []

class ACO:
    def __init__(self, num, model, data):
        self.num = num
        self.model = model
        self.data= pd.read_csv(data)

    def gerarXy(self):
        self.X = self.dados.iloc[:, :-1].values 
        self.y = self.dados.iloc[:, -1].values 
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(self.X, self.y, test_size=0.3, random_state=1)

    def avaliacao(self, trajeto):
        if(self.model == 0):
            tradutor = {7:"gini", 8:"entropy", 9:"log_loss"}
            arvore = DecisionTreeClassifier(max_depth= int(trajeto[0]), criterion= tradutor[trajeto[1]], min_samples_split= trajeto[2])
            arvore.fit(self.X_treino, self.y_treino)
            previsao = arvore.predict(self.X_teste)
            acuracia = metrics.accuracy_score(self.y_teste, previsao)    
        return acuracia
    
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
    
    def calcularFerormonio(self, distancia, nFormiga):
        w = (1/(1 + e**(-distancia + 0.45)) + 0.55)**17 + 3 * log(nFormiga, 10)
        return w

    def calcularRotaDecimais(self, vMax, vMin, linha, formiga):#indices = lista dos vertices(colunas)
        z = 0 #variavel para acessar cada coluna da linha
        a = 0 #= para atualizar o ponto de partida do index e colocar os indices de forma correta
        v = 0 #= para indicar que saiu do primeiro ciclo
        u = 0 #= para indicar que chegou no vértice de saída
        t = 0 #= Para utilizar o metodo de criar listas da matadj apenas uma vez
        r = 0 #= Para ajudar na valorização do atributo
        soma = 0
        vAtual = vMax/2
        escolhido = 0
        caminho = []
        indices = []
        iS = []
        linhaPercorrida = [] # Conjunto de cada aresta percorrida de cada vértice
        possibilidades = []
        tradutor = {2:0, 3:1, 4:2, 5:3, 6:4, 11:0, 12:1, 13:2, 14:3, 15:4}
        while(v != 1 or u != 1):
            if(v == 1 and len(caminho)< 2 or v==0):
                i = list(self.matadj[linha])
            else:
                i = iS[tradutor[linha]]           
            while(z != 16):
                if(v == 0):
                    linhaPercorrida.append(0)
                if(i[z] == 1):
                    indices.append(i.index(1, a))
                    a = indices[-1] + 1
                z+=1
            if(linhaPercorrida[linha] == 0 and v == 1):
                for ind in indices:
                    percorridos.append(0)
                linhaPercorrida.pop(linha)              
                linhaPercorrida.insert(linha, percorridos)    
            if(len(caminho) > 1):
                vertice = linhaPercorrida[caminho[-2]]
                vertice[tradutor[linha]] += 1
            print(linhaPercorrida)    
            percorridos = []
            if(iS != []):
                if(vertice[tradutor[linha]] == 3):
                    i[linha] = 0
                    indices.remove(linha)
            while(vMin >= vAtual or vAtual >= vMax or r == 0):#PAREI AQUI ADICIONAR A VALORIZAÇÃO DO ATRIBUTO NESSE WHILE              
                for ind in indices:
                    soma +=  (self.matFer[linha][ind]**formiga.alpha) * (self.matPos[linha][ind]**formiga.beta)
                for ind in indices:
                    possibilidades.append(((self.matFer[linha][ind]**formiga.alpha) * (self.matPos[linha][ind]**formiga.beta))/soma)        
                escolhido = np.random.choice(indices, p=possibilidades)
                r+=1
            caminho.append(escolhido)
            linha = caminho[-1]
            if(t == 0 and v == 1):
                for i in indices:
                    iS.append(list(self.matadj[i])) 
                t+=1
            if(escolhido == indices[-1] and v == 1):
                u = 1
            else:
                indices.clear()   
            possibilidades.clear()
            z = 0
            a = 0
            soma = 0
            r = 0
            print(escolhido)
            if(v == 0):
                v+=1
        return None #uma lista contendo o caminho e o valor final do atributo       

        



        
            

aco = ACO(10, 0, "heart.csv")
aco.matrizAdj()
aco.calcularRotaDecimais(150, 3, 1, aco.gerarFormiga())
