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
import random

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

def entropia_apriori(probabilidades):
    entropia = -np.sum(probabilidades * np.log2(probabilidades))
    return entropia

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
    entropiaB = (-pbj0 * np.log2(pbj0)) + (- pbj1 * np.log2(pbj1))
    
    # P(ai/bj)
    pa0b0 = matriz_canal[0][0] * p_fuente[0] / pbj0
    pa0b1 = matriz_canal[0][1] * p_fuente[0] / pbj1

    pa1b0 = 1 - pa0b0
    pa1b1 = 1 - pa0b1
    
    # Susceso ==> P(ai,bj)
    suceso00 = matriz_canal[0][0] * p_fuente[0]
    suceso01 = matriz_canal[0][1] * p_fuente[0]
    suceso10 = matriz_canal[1][0] * p_fuente[1]
    suceso11 = matriz_canal[1][1] * p_fuente[1]
    
    # Entropias a posteriori H(A,bj)
    entropiab0 = (-pa0b0 * np.log2(pa0b0)) + (- pa1b0 * np.log2(pa1b0))
    entropiab1 = (-pa0b1 * np.log2(pa0b1)) + (- pa1b1 * np.log2(pa1b1))
    
    # Equivocacion H(A/B)
    equivocacion = (pbj0 * entropiab0) + (pbj1 * entropiab1) 
    
    # Informacion mutua I(A,B)
    informacion_mutua = entropiaA - equivocacion
    
    # Entropia afin H(A,B)
    entropia_afin = entropiaB + equivocacion
    
    # Perdida H(B/A)
    perdida = entropia_afin - entropiaA
    
    return entropiaA, entropiaB, entropiab0, entropiab1, equivocacion, informacion_mutua, entropia_afin, perdida

def aplicar_parity_check(mensaje):
    # Aplicar el método de paridad cruzada al mensaje
    mensaje_list = list(mensaje)
    paridad = 0
    for bit in mensaje_list:
        paridad ^= int(bit)
    return mensaje + str(paridad)

def generar_mensajes_binarios(N, M):
    mensajes = []
    for _ in range(N):
        mensaje = ''.join(random.choice('01') for _ in range(M))
        mensajes.append(mensaje)
    return mensajes

def main():
    filename = "tp4_sample0.txt"
    
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
    
    if "-p" in sys.argv:
        mensajes = generar_mensajes_binarios(3, 10)
        mensajes_con_parity = [aplicar_parity_check(mensaje) for mensaje in mensajes]
        
        # Simular la transmisión y contar los mensajes enviados correctamente, incorrectos y corregidos
        mensajes_correctos = 0
        mensajes_incorrectos = 0
        mensajes_corregidos = 0
        
        for mensaje, mensaje_con_parity in zip(mensajes, mensajes_con_parity):
            # Simular un error en la transmisión (puedes cambiar esto según tus necesidades)
            error = random.choice([True, False])
            if error:
                mensajes_incorrectos += 1
                # Corregir el error invirtiendo el último bit
                mensaje_recibido = mensaje_con_parity[:-1] + str(1 - int(mensaje_con_parity[-1]))
            else:
                mensajes_correctos += 1
                mensaje_recibido = mensaje_con_parity
            
            if mensaje_recibido == mensaje:
                mensajes_corregidos += 1
        
        print("Resultados después de la transmisión:")
        print(f"Mensajes enviados correctamente: {mensajes_correctos}")
        print(f"Mensajes incorrectos: {mensajes_incorrectos}")
        print(f"Mensajes corregidos: {mensajes_corregidos}")
    
    return 0

if __name__ == "__main__":
    main()



