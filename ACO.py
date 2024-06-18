import numpy as np
import random as rd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from math import e
from math import log


class Formiga:
    def __init__(self):
        self.alpha = rd.randint(1,3)

class ACO:
    def __init__(self, num, model, data):
        self.num = num
        self.model = model
        self.dados= pd.read_csv(data)

    def gerarXy(self):
        self.X = self.dados.iloc[:, :-1].values 
        self.y = self.dados.iloc[:, -1].values 
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(self.X, self.y, test_size=0.3, random_state=1)

    def avaliacao(self, trajeto):
        if(self.model == 0):
            tradutor = {0:"gini", 1:"entropy", 2:"log_loss"}
            arvore = DecisionTreeClassifier(max_depth= int(trajeto[0]), criterion= tradutor[trajeto[1]], min_samples_split= trajeto[2])
            arvore.fit(self.X_treino, self.y_treino)
            previsao = arvore.predict(self.X_teste)
            acuracia = metrics.accuracy_score(self.y_teste, previsao)
        elif(self.model == 1):
            tradutor = {0:"uniform", 1:"distance"}
            knn = KNeighborsClassifier(n_neighbors=trajeto[0], weights=tradutor[trajeto[1]])
            knn.fit(self.X_treino, self.y_treino)
            previsao = knn.predict(self.X_teste)
            acuracia = metrics.accuracy_score(self.y_teste, previsao)
        elif(self.model == 2):
            tradutor = {0:"gini", 1:"entropy", 2:"log_loss"}
            floresta = RandomForestClassifier(n_estimators=int(trajeto[0]), max_depth=int(trajeto[1]), criterion=tradutor[trajeto[2]], min_samples_split=trajeto[3])
            floresta.fit(self.X_treino, self.y_treino)
            previsao = floresta.predict(self.X_teste)
            self.acuracia1 = metrics.accuracy_score(self.y_teste, previsao)
        return acuracia
    
    def matrizAdj(self):
        if(self.model == 0):
            self.matadj = np.zeros((16, 16), dtype=np.float64)
            self.matadj[0, 1], self.matadj[1, 2], self.matadj[1, 3], self.matadj[1, 4], self.matadj[1,5], self.matadj[2, 2], self.matadj[2, 3], self.matadj[2, 4], self.matadj[2, 5], self.matadj[2, 6], self.matadj[3,3], self.matadj[3,4], self.matadj[3,5], self.matadj[3,6], self.matadj[3, 2], self.matadj[4, 2], self.matadj[4, 3], self.matadj[4,4], self.matadj[4,5], self.matadj[4,6], self.matadj[5,2], self.matadj[5,3], self.matadj[5,4], self.matadj[5,5], self.matadj[5,6], self.matadj[6,7], self.matadj[6,8], self.matadj[6,9], self.matadj[7,10], self.matadj[8,10], self.matadj[9,10], self.matadj[10,11], self.matadj[10,12], self.matadj[10,13], self.matadj[10,14], self.matadj[11,11], self.matadj[11,12], self.matadj[11,13], self.matadj[11,14], self.matadj[11,15], self.matadj[12,11], self.matadj[12,12], self.matadj[12,13], self.matadj[12,14], self.matadj[12,15], self.matadj[13,11], self.matadj[13,12], self.matadj[13,13], self.matadj[13,14], self.matadj[13,15], self.matadj[14,11], self.matadj[14,12], self.matadj[14,13], self.matadj[14,14], self.matadj[14,15] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            self.matrizFer()
        elif(self.model == 1):
            self.matadj = np.zeros((8, 8), dtype=np.float64)                           
            self.matadj[0,1], self.matadj[0,2], self.matadj[0,3], self.matadj[0,4], self.matadj[1,5], self.matadj[1,6], self.matadj[2,5], self.matadj[2,6], self.matadj[3,5], self.matadj[3,6], self.matadj[4,5], self.matadj[4,6], self.matadj[5,7], self.matadj[6,7] = 1,1,1,1,1,1,1,1,1,1,1,1,1,1
            self.matrizFer()
        elif(self.model == 2):
            self.matadj = np.zeros((22, 22), dtype=np.float64)
            self.matadj[0,1], self.matadj[1,2], self.matadj[1,3], self.matadj[1,4], self.matadj[1,5], self.matadj[2,2], self.matadj[2,3], self.matadj[2,4], self.matadj[2,5], self.matadj[2,6], self.matadj[3,2], self.matadj[3,3], self.matadj[3,4], self.matadj[3,5], self.matadj[3,6], self.matadj[4,2], self.matadj[4,3], self.matadj[4,4], self.matadj[4,5], self.matadj[4,6], self.matadj[5,2], self.matadj[5,3], self.matadj[5,4], self.matadj[5,5], self.matadj[5,6], self.matadj[6,7], self.matadj[7,8], self.matadj[7,9], self.matadj[7,10], self.matadj[7,11], self.matadj[8,8], self.matadj[8,9], self.matadj[8,10], self.matadj[8,11],self.matadj[8,12], self.matadj[9,8], self.matadj[9,9], self.matadj[9,10], self.matadj[9,11], self.matadj[9,12], self.matadj[10,8], self.matadj[10,9], self.matadj[10,10], self.matadj[10,11], self.matadj[10,12], self.matadj[11,8], self.matadj[11,9], self.matadj[11,10], self.matadj[11,11], self.matadj[11,12], self.matadj[12,13], self.matadj[12,14], self.matadj[12,15], self.matadj[13,16], self.matadj[14,16], self.matadj[15,16], self.matadj[16,17], self.matadj[16,18], self.matadj[16,19], self.matadj[16,20], self.matadj[17,17], self.matadj[17,18], self.matadj[17,19], self.matadj[17,20], self.matadj[17,21], self.matadj[18,17], self.matadj[18,18], self.matadj[18,19], self.matadj[18,20], self.matadj[18,21], self.matadj[19,17], self.matadj[19,18], self.matadj[19,19], self.matadj[19,20], self.matadj[19,21], self.matadj[20,17], self.matadj[20,18], self.matadj[20,19], self.matadj[20,20], self.matadj[20,21] = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            self.matrizFer()   
    def matrizFer(self):
        if(self.model == 0):
            self.matFer = np.zeros((16, 16), dtype=np.float64)
            for i in range(len(self.matadj)):
                for j in range(len(self.matadj)):
                    if(self.matadj[i,j] == 1):
                        self.matFer[i, j] = 5 
        elif(self.model == 1):
            self.matFer = np.zeros((8, 8), dtype=np.float64)
            for i in range(len(self.matadj)):
                for j in range(len(self.matadj)):
                    if(self.matadj[i,j] == 1):
                        self.matFer[i, j] = 5
        elif(self.model == 2):
            self.matFer = np.zeros((22, 22), dtype=np.float64)
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
    
    def formigasBobinhas(self):
        i = 0
        caminho = []
        caminhosRealizados = []
        formigasEscolhas = []
        acuracias = []
        atributos = []
        ferormonios = []
        if(self.model == 0):
            while(i != 10):
                f = self.gerarFormiga()
                caminho.append(0)
                caminho.append(1)
                lista1 = self.calcularRotaDecimais(150, 3, caminho[-1], f)
                atributos.append(lista1[1])
                subLista1 = lista1[0]
                for cam in subLista1:
                    caminho.append(cam)
                lista2 = self.calcularRotasNormais([0,1,2], f, caminho[-1])
                atributos.append(lista2[1])
                caminho.append(lista2[0])
                caminho.append(10)
                lista3 = self.calcularRotaDecimais(0.9, 0.1, caminho[-1], f)
                atributos.append(lista3[1])
                subLista3 = lista3[0]
                for cam in subLista3:
                    caminho.append(cam)    
                acuracias.append(self.avaliacao(atributos))
                formigasEscolhas.append(str(i) + "-->")
                formigasEscolhas.append(atributos)
                caminhosRealizados.append(caminho)
                caminho = []
                atributos = []
                lista1 = []
                lista2 = []
                lista3 = []
                subLista1 = []
                subLista3 = []
                ferormonios.append(int(self.calcularFerormonio(acuracias[i], i+1)))
                i+=1
            i = 0   
            while(i != 10):
                k=1
                for j in caminhosRealizados[i]:
                    if(j != 15):
                        self.matFer[j][caminhosRealizados[i][k]] += ferormonios[i]
                        self.matFer[j][caminhosRealizados[i][k]] = self.matFer[j][caminhosRealizados[i][k]] * 0.9
                        k+=1
                i+=1
        elif(self.model == 1):
            while(i != 10):
                f = self.gerarFormiga()
                caminho.append(0)
                lista1 = self.calcularRotasNormais([3, 5, 7, 11], f, caminho[-1])
                atributos.append(lista1[1])
                caminho.append(lista1[0])
                lista2 = self.calcularRotasNormais([0, 1], f, caminho[-1])
                atributos.append(lista2[1])
                caminho.append(lista2[0])
                caminho.append(7)
                acuracias.append(self.avaliacao(atributos))
                formigasEscolhas.append(str(i) + "-->")
                formigasEscolhas.append(atributos)
                caminhosRealizados.append(caminho)
                caminho = []
                atributos = []
                lista1 = []
                lista2 = []
                ferormonios.append(int(self.calcularFerormonio(acuracias[i], i+1)))
                i+=1
            i = 0   
            while(i != 10):
                k=1
                for j in caminhosRealizados[i]:
                    if(j != 7):
                        self.matFer[j][caminhosRealizados[i][k]] += ferormonios[i]
                        self.matFer[j][caminhosRealizados[i][k]] = self.matFer[j][caminhosRealizados[i][k]] * 0.9
                        k+=1
                i+=1
            print(self.matFer)                                            

    def calcularRotaDecimais(self, vMax, vMin, linha, formiga):
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
            percorridos = []
            if(iS != []):
                if(vertice[tradutor[linha]] == 3):
                    iS[tradutor[caminho[-2]]][linha] = 0
            while(vMin >= vAtual or vAtual >= vMax or r == 0):
                if(r == 1):
                    vAtual = antigo            
                for ind in indices:
                    soma +=  (self.matFer[linha][ind]**formiga.alpha)
                for ind in indices:
                    possibilidades.append(((self.matFer[linha][ind]**formiga.alpha))/soma)        
                escolhido = np.random.choice(indices, p=possibilidades)
                if(escolhido == 2 or escolhido == 11):
                    antigo = vAtual
                    vAtual += -vMax/3
                elif(escolhido == 3 or escolhido == 12):
                    antigo = vAtual
                    vAtual += -vMax/5
                elif(escolhido == 4 or escolhido == 13):
                    antigo = vAtual
                    vAtual += vMax/7
                elif(escolhido == 5 or escolhido == 14):
                    antigo = vAtual
                    vAtual += vMax/11
                if(r == 0):    
                    r+=1
                soma = 0
                possibilidades.clear()                  
            caminho.append(escolhido)
            linha = caminho[-1]
            if(t == 0 and v == 1):
                for i in indices:
                    if(i != 6):
                        iS.append(list(self.matadj[i])) 
                t+=1
            if(escolhido == indices[-1] and v == 1):
                u = 1
            else:
                indices.clear()   
            z = 0
            a = 0
            r = 0
            if(v == 0):
                v+=1
            if(u == 1):
                retorno = [caminho, vAtual]       
        return retorno #uma lista contendo o caminho e o valor final do atributo

    def calcularRotasNormais(self, atributos, formiga, linha): #atributos: lista com as possiveis elementos a serem escolhidos 0(gini), 1(entropy), ... 
        i = list(self.matadj[linha])
        k = 0
        soma = 0
        indices = []
        possibilidades = []
        for j in i:
            if(j == 1):
                indices.append(i.index(1, k))
                k = indices[-1] + 1
        for ind in indices:
            soma += (self.matFer[linha][ind]**formiga.alpha)
        for ind in indices:
            possibilidades.append(((self.matFer[linha][ind]**formiga.alpha))/soma)
        escolhido = np.random.choice(indices, p=possibilidades)
        if(escolhido == indices[0]):
            retorno = [escolhido, atributos[0]]
            return retorno
        elif(escolhido == indices[1]):
            retorno = [escolhido, atributos[1]]
            return retorno  
        elif(escolhido == indices[2]):
            retorno = [escolhido, atributos[2]]
            return retorno
        elif(escolhido == indices[3]):
            retorno = [escolhido, atributos[3]]
            return retorno        

    def caminharFormmiga(self):
        caminho = []
        caminhosRealizados = []
        formigasEscolhas = []
        acuracias = []
        i = 10
        atributos = []
        if(self.model == 0):
            self.formigasBobinhas()
            while(self.num != i):
                f = self.gerarFormiga()
                caminho.append(0)
                caminho.append(1)
                lista1 = self.calcularRotaDecimais(150, 3, caminho[-1], f)
                atributos.append(lista1[1])
                subLista1 = lista1[0]
                for cam in subLista1:
                    caminho.append(cam)
                lista2 = self.calcularRotasNormais([0,1,2], f, caminho[-1])
                atributos.append(lista2[1])
                caminho.append(lista2[0])
                caminho.append(10)
                lista3 = self.calcularRotaDecimais(0.9, 0.1, caminho[-1], f)
                atributos.append(lista3[1])
                subLista3 = lista3[0]
                for cam in subLista3:
                    caminho.append(cam)    
                acuracias.append(self.avaliacao(atributos))
                formigasEscolhas.append(str(i) + "-->")
                formigasEscolhas.append(atributos)
                caminhosRealizados.append(caminho)
                w = self.calcularFerormonio(acuracias[-1], i)
                k=1
                for j in caminho:
                    if(j != 15):
                        self.matFer[j][caminho[k]] += w
                        self.matFer[j][caminho[k]] = self.matFer[j][caminho[k]] * 0.9
                        k+=1
                caminho = []
                atributos = []
                lista1 = []
                lista2 = []
                lista3 = []
                subLista1 = []
                subLista3 = []
                i+=1
        elif(self.model == 1):
            self.formigasBobinhas()
            while(i != self.num):
                f = self.gerarFormiga()
                caminho.append(0)
                lista1 = self.calcularRotasNormais([3, 5, 7, 11], f, caminho[-1])
                atributos.append(lista1[1])
                caminho.append(lista1[0])
                lista2 = self.calcularRotasNormais([0, 1], f, caminho[-1])
                atributos.append(lista2[1])
                caminho.append(lista2[0])
                caminho.append(7)
                acuracias.append(self.avaliacao(atributos))
                formigasEscolhas.append(str(i) + "-->")
                formigasEscolhas.append(atributos)
                caminhosRealizados.append(caminho)
                w = self.calcularFerormonio(acuracias[-1], i)
                k=1
                for j in caminho:
                    if(j != 7):
                        self.matFer[j][caminho[k]] += w
                        self.matFer[j][caminho[k]] = self.matFer[j][caminho[k]] * 0.9
                        k+=1
                caminho = []
                atributos = []
                lista1 = []
                lista2 = []
                i+=1           
                 

        
#FINALIZAR FLORESTA


        
            

aco = ACO(50, 2, "heart.csv")
aco.matrizAdj()
aco.gerarXy()
aco.caminharFormmiga()
