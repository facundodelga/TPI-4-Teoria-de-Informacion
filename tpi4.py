# Desarrollar un programa que pueda ser ejecutado por consola del siguiente modo:

# tpi4 probs.txt N M [-p]

# Donde:

# tpi4 es el programa ejecutable
# probs.txt es un archivo de texto ASCII
# N y M son números naturales
# -p es un flag opcional
# El programa debe realizar las siguientes acciones:

# Leer del archivo probs.txt las probabilidades de la fuente binaria (primera línea) y la matriz del canal binario (segunda y tercera línea).
# Calcular las entropías del canal, la equivocación y la información mutua.
# Simular el envío de N mensajes aleatorios de longitud M.
# Si se incluye el flag -p, aplicar el método de paridad cruzada al conjunto de mensajes a enviar.
# Informar la cantidad de mensajes enviados correctamente, la cantidad de mensajes erróneos y la cantidad de mensajes corregidos.
import sys
import numpy as np
from random import random

def leer_probabilidades(archivo):
    matriz_canal = []
    
    with open(archivo, "r") as f:
        probabilidades = f.readline()
        probabilidades = probabilidades.split(" ")
        
        probabilidades = [float(probabilidad) for probabilidad in probabilidades]
        
        fuente = probabilidades
        
        for linea in f:
            probabilidades = linea.split(" ")
            matriz_canal.append([float(probabilidad) for probabilidad in probabilidades])
        
        matriz_canal = np.array(matriz_canal)
    
    return fuente, matriz_canal

def info(p):
    if p > 0:
        return - np.log2(p)
    else:
        return 0

def entropia_apriori(probabilidades):
    return np.sum([p * info(p) for p in probabilidades])

def probBj(p_fuente, p_canal):
    b = np.dot(p_fuente, p_canal)
    return b

def calculos(p_fuente, matriz_canal):
    # Calcular las entropías del canal, la equivocación y la información mutua.
    
    # H(A)
    entropiaA = entropia_apriori(p_fuente)
    
    # P(bj)
    pbj0 = probBj(p_fuente, matriz_canal[:,0])
    pbj1 = probBj(p_fuente, matriz_canal[:,1])
    
    # H(B)
    entropiaB = pbj0 * info(pbj0) + pbj1 * info(pbj1)
    
    # P(ai/bj)
    if pbj0 != 0:
        pa0b0 = matriz_canal[0][0] * p_fuente[0] / pbj0
        pa0b1 = matriz_canal[0][1] * p_fuente[0] / pbj1
    else:
        pa0b0 = 0
        pa0b1 = 0   

    pa1b0 = 1 - pa0b0
    pa1b1 = 1 - pa0b1
    
    # Susceso ==> P(ai,bj)
    suceso00 = matriz_canal[0][0] * p_fuente[0]
    suceso01 = matriz_canal[0][1] * p_fuente[0]
    suceso10 = matriz_canal[1][0] * p_fuente[1]
    suceso11 = matriz_canal[1][1] * p_fuente[1]
    
    # Entropias a posteriori H(A,bj)
    entropiab0 = pa0b0 * info(pa0b0) + pa1b0 * info(pa1b0)
    entropiab1 = pa0b1 * info(pa0b1) + pa1b1 * info(pa1b1)
    
    # Equivocacion H(A/B)
    equivocacion = (pbj0 * entropiab0) + (pbj1 * entropiab1) 
    
    # Informacion mutua I(A,B)
    informacion_mutua = entropiaA - equivocacion
    
    # Entropia afin H(A,B)
    entropia_afin = entropiaB + equivocacion
    
    # Perdida H(B/A)
    perdida = entropia_afin - entropiaA
    
    return entropiaA, entropiaB, entropiab0, entropiab1, equivocacion, informacion_mutua, entropia_afin, perdida
#################
# Simular el envío de N mensajes aleatorios de longitud M.
# Si se incluye el flag -p, aplicar el método de paridad cruzada al conjunto de mensajes a enviar.

def generaMsj(N, M, fuente):
    matMsj = np.zeros((N, M), dtype=int)
    for i in range(N):
        for j in range(M):
            prob = random()
            if prob > 0 and prob <= fuente[0]:
                matMsj[i][j] = 0
            else:
                matMsj[i][j] = 1
    return matMsj

def aplicarParidad(mat, N, M):
    matP = np.zeros((N+1, M+1), dtype=int)

    #copiar el contenido de la matriz mat en la matriz matP
    for i in range(N):
        for j in range(M):
            matP[i][j] = mat[i][j]
    
    #calcul0 la paridad de cada fila
    for i in range(N):
        paridad = 0
        for j in range(M):
            paridad = paridad ^ mat[i][j]
        matP[i][M] = paridad
    
    #calculo la paridad de cada columna
    for j in range(M):
        paridad = 0
        for i in range(N):
            paridad = paridad ^ mat[i][j]
        matP[N][j] = paridad
    
    #calculo la paridad de la matriz
    p1 = 0
    p2 = 0
    for i in range(N):
        p1 = p1 ^ matP[i][M]
    for j in range(M):  
        p2 = p2 ^ matP[N][j]
    
    if p1 == p2:
        matP[N][M] = p1

    return matP



def simularEnvio(matriz_canal, mat):
    #mat = generaMsj(N, M, fuente)
    matE = np.array(mat)
    #if p == true: #  "-p":
    #    mat = aplicarParidad(mat, N, M)
    
    #cada bit lo paso por el canal
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            aux = mat[i][j]
            prob = random()
            if aux == 0: #uso la primer fila de la matriz de canal
                if prob > 0 and prob <= matriz_canal[0][0]:
                    matE[i][j] = 0
                else:
                    matE[i][j] = 1
            else: #segunda fila
                if prob > 0 and prob <= matriz_canal[1][0]:
                    matE[i][j] = 0
                else:
                    matE[i][j] = 1
    return matE


def simular(p_fuente, matriz_canal, N, M, p):
    mat = generaMsj(N, M, p_fuente) #original sin pasarla por el canal

    if p: #  "-p":
        mat = aplicarParidad(mat, N, M)# matriz original sin pasarla por la fuente pero con flag -p
    
    matE = simularEnvio(matriz_canal, mat) #matriz enviada por el canal

    return mat, matE

def mostrarMatriz(mat):
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            print(mat[i][j], end=" ")
        print("")
    print("")

# Informar la cantidad de mensajes enviados correctamente, la cantidad de mensajes erróneos y la cantidad de mensajes corregidos.
def cantMsjCorrectosEIncorrectos(mat, matE, N, M):
    cant = 0
    for i in range(N):
        j = 0
        while j < M and mat[i][j] == matE[i][j]:
            j += 1
        if j == M:    
            cant += 1

    return cant, N - cant

def cantMsjCorregidos(mat):
    auxFila = 0
    auxColumna = 0
    #calcul0 la paridad de cada fila
    for i in range(len(mat)):
        paridad = 0
        for j in range(len(mat[i])):
            paridad = paridad ^ mat[i][j]

        if paridad != 0:
            if auxFila == 0:
                auxFila = i + 1
            else:
                auxFila = 0
                break
    
    #calculo la paridad de cada columna
    for j in range(len(mat[0])):
        paridad = 0
        for i in range(len(mat)):
            paridad = paridad ^ mat[i][j]

        if paridad != 0:
            if auxColumna == 0:
                auxColumna = j + 1
            else:
                auxColumna = 0
                break
    
    print("auxFila: ", auxFila)
    print("auxColumna: ", auxColumna)
    if auxFila > 0 and auxColumna > 0 and auxFila < len(mat) and auxColumna < len(mat[0]):
        return 1
    else:
        return 0

def main():
    #filename = "tp4_sample0.txt"
    filename = sys.argv[1]
    N = int(sys.argv[2])
    M = int(sys.argv[3])

    p = len(sys.argv) > 4 and sys.argv[4] == "-p"
    
    p_fuente, matriz_canal = leer_probabilidades(filename)
    HA, HB, HB0, HB1, equivocacion, informacion_mutua, HAfin, perdida = calculos(p_fuente, matriz_canal)
    
    print("")
    print(f"Entropia a priori H(A): {HA}")
    print(f"Entropia a priori H(B): {HB}")
    print(f"Entropia a posteriori H(A/b=0): {HB0}")
    print(f"Entropia a posteriori H(A/b=1): {HB1}")
    print(f"Equivocacion del canal H(A/B): {equivocacion}")
    print(f"Informacion mutua I(A,B): {informacion_mutua}")
    print(f"Entropia afin H(A,B): {HAfin}")
    print(f"Perdida H(B/A): {perdida}")
    print("")
    
   
    matOriginal, matEnviada = simular(p_fuente, matriz_canal, N, M, p)

    mostrarMatriz(matOriginal)
    mostrarMatriz(matEnviada)
     #calculo la cantidad de mensajes enviados correctamente, la cantidad de mensajes erróneos
    cantC, cantI = cantMsjCorrectosEIncorrectos(matOriginal, matEnviada, N, M)
    print("Cantidad de msj enviados correctamente: ", cantC)
    print("Cantidad de msj enviados erroneos: ", cantI)
   
    cantCorregidos = cantMsjCorregidos(matEnviada) if p else 0
    if p:
        print("Cantidad de msj corregidos: ", cantCorregidos)
    
    return 0

if __name__ == "__main__":
    main()



