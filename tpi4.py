import numpy as np
import sys  # Se agrega la importación del módulo sys

# Función para leer las probabilidades desde un archivo
def leer_probabilidades(nombre_archivo):
    with open(nombre_archivo) as f:
        # Lee las probabilidades de la fuente
        p_fuente = np.array([float(x) for x in f.readline().split()])
        
        # Lee las probabilidades del canal como una matriz
        p_canal = np.array([
            [float(x) for x in fila.split()]
            for fila in f.readlines()
        ])
    return p_fuente, p_canal

# Función para calcular la entropía de una distribución de probabilidad
def calcular_entropia(p):
    # La entropía se calcula como la suma ponderada de las probabilidades logarítmicas negativas
    entropia = -np.sum(p * np.log2(p))
    return entropia

# Función para calcular la equivocación entre la fuente y el canal
def calcular_equivocacion(p_fuente, p_canal):
    # Calcula la probabilidad conjunta entre la fuente y el canal
    p_conjunta = np.einsum('i,jk->ik', p_fuente, p_canal)
    
    # Calcula la probabilidad marginal del canal (ruido)
    p_ruido = np.sum(p_canal, axis=0)
    
    # Calcula la equivocación utilizando la fórmula específica para el canal discreto
    equivocacion = calcular_entropia(p_ruido) - np.sum(p_conjunta * np.log2(p_canal / p_ruido))
    return equivocacion

# Función para calcular la información mutua entre la fuente y el canal
def calcular_informacion_mutua(p_fuente, p_canal):
    # La información mutua se calcula como la diferencia entre la entropía de la fuente y la equivocación
    informacion_mutua = calcular_entropia(p_fuente) - calcular_equivocacion(p_fuente, p_canal)
    return informacion_mutua

# Función para simular la transmisión de mensajes a través de un canal
def simular_transmision(p_fuente, p_canal, N, M, paridad_cruzada=False):
    if paridad_cruzada:
        mensajes = generar_mensajes_paridad_cruzada(N, M)
    else:
        mensajes = generar_mensajes_aleatorios(N, M)

    # Simula la transmisión de mensajes a través del canal
    mensajes_transmitidos = np.random.choice(2, size=(N, M), p=p_canal)
    mensajes_recibidos = mensajes_transmitidos.dot(p_canal.T)

    # Calcula la cantidad de mensajes correctos e incorrectos
    mensajes_correctos = np.sum(mensajes == mensajes_recibidos)
    mensajes_incorrectos = N - mensajes_correctos

    if paridad_cruzada:
        mensajes_corregidos = corregir_mensajes_paridad_cruzada(mensajes_recibidos)
    else:
        mensajes_corregidos = 0

    return mensajes_correctos, mensajes_incorrectos, mensajes_corregidos

# Función para generar mensajes aleatorios
def generar_mensajes_aleatorios(N, M):
    return np.random.choice(2, size=(N, M))

# Función para generar mensajes con paridad cruzada
def generar_mensajes_paridad_cruzada(N, M):
    mensajes = generar_mensajes_aleatorios(N, M - 1)

    # Calcula los bits de paridad y los agrega a los mensajes
    bits_paridad = np.sum(mensajes, axis=1) % 2
    bits_paridad = np.expand_dims(bits_paridad, axis=1)

    mensajes = np.concatenate((mensajes, bits_paridad), axis=1)

    return mensajes

# Función para corregir mensajes con paridad cruzada
def corregir_mensajes_paridad_cruzada(mensajes_recibidos):
    bits_paridad = mensajes_recibidos[:, -1]
    mensajes_corregidos = mensajes_recibidos[:, :-1]

    # Calcula los bits de paridad esperados
    bits_paridad_reales = np.sum(mensajes_corregidos, axis=1) % 2

    # Identifica y corrige errores en los bits de paridad
    errores = np.bitwise_xor(bits_paridad, bits_paridad_reales)

    for i, error in enumerate(errores):
        if error:
            for j in range(M - 1):
                if mensajes_recibidos[i, j] == 1:
                    mensajes_corregidos[i, j] = 0
                    break

    # Retorna la cantidad de mensajes corregidos
    return np.sum(mensajes_corregidos == mensajes)

# Función principal del programa
def principal():
    # Lee las probabilidades desde el archivo
    p_fuente, p_canal = leer_probabilidades('probs.txt')

    # Calcula la entropía del canal, la equivocación y la información mutua
    entropia_canal = calcular_entropia(p_canal)
    equivocacion = calcular_equivocacion(p_fuente, p_canal)
    informacion_mutua = calcular_informacion_mutua(p_fuente, p_canal)

    # Imprime los resultados
    print('Entropía del canal:', entropia_canal)
    print('Equivocación:', equivocacion)
    print('Información mutua:', informacion_mutua)

    # Solicita al usuario ingresar el número de mensajes y la longitud del mensaje
    N = int(input('Ingrese el número de mensajes (N): '))
    M = int(input('Ingrese la longitud del mensaje (M): '))

    # Verifica si se debe usar paridad cruzada según los argumentos de línea de comandos
    if '-p' in sys.argv:
        paridad_cruzada = True
    else:
        paridad_cruzada = False

    # Realiza la simulación de la transmisión y obtiene los resultados
    mensajes_correctos, mensajes_incorrectos, mensajes_corregidos = simular_transmision(p_fuente, p_canal, N, M, paridad_cruzada)

    # Imprime los resultados de la simulación
    print('Mensajes correctos:', mensajes_correctos)
    print('Mensajes incorrectos:', mensajes_incorrectos)
    print('Mensajes corregidos (paridad cruzada):', mensajes_corregidos)


if __name__ == '__main__':
    principal()
