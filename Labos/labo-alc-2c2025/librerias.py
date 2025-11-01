import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def esCuadrada(A):
    filas,columnas = A.shape
    return filas == columnas

matrizCuadrada = np.array([[2,1,8],[4,5,9],[10,5,6]])
matrizSimetrica = np.array([[0,1,1],[1,0,1],[1,1,0]])
A2 = matrizCuadrada.copy()


def triangSup(A):
    copia = A.copy()
    filas,columnas = A.shape
    for i in range(filas):
        for j in range(columnas):
            if j<=i:
                copia[i][j] = 0.0
    return copia


def triangInf(A):
    copia = A.copy()
    filas,columnas = A.shape
    for i in range(filas):
        for j in range(columnas):
            if j>=i:
                copia[i][j] = 0.0
    return copia    


def diagonal(A):
    filas, columnas = A.shape
    tamaño = min(filas, columnas)
    res = np.zeros(A.shape)
    for i in range(tamaño):
        res[i][i] = A[i][i]
    return res

def traza(A):
    filas, columnas = A.shape
    tamaño = min(filas, columnas)
    suma = 0
    for i in range(tamaño):
        suma += A[i][i]
    return suma    

def traspuesta(A):
    filas, columnas = A.shape
    res = np.zeros((columnas, filas))
    for i in range(filas):
        for j in range(columnas):
            res[j][i] = A[i][j]
    return res 


def sonIguales(A,B,tol=1e-5):
    if A.shape != B.shape:
        return False
    else:
        filas, columnas = A.shape
        for i in range(filas):
            for j in range(columnas):
                if abs(A[i][j] - B[i][j]) > tol:
                    return False
        return True
        


def esSimetrica(A):
    if not esCuadrada(A):
        return False
    else:
        return sonIguales(A, traspuesta(A))
    
def productoEscalar(a,b):
    tamaño = a.shape[0]
    suma = 0
    for i in range(tamaño):
        suma += a[i] * b[i]
    return suma


def calcularAx(A, x):
    res = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        res[i] = productoEscalar(A[i],x)
    return res



def intercambiarFilas(A,i,j):
    temporal = A[i].copy()
    A[i] = A[j]
    A[j] = temporal


def sumaVectores(a,b):
    tamaño = a.shape[0]
    res = np.zeros(tamaño)
    for i in range(tamaño):
        res[i] = a[i] + b[i]
    return res


def vectorPorEscalar(a, c):
    tamaño = a.shape[0]
    res = np.zeros(tamaño)
    for i in range(tamaño):
        res[i] = a[i] * c
    return res


def sumarFilaMultiplo(A,i,j,s):
    multiplo = vectorPorEscalar(A[j], s)
    nueva_fila = sumaVectores(multiplo, A[i])
    A[i] = nueva_fila



def esDiagonalmenteDominante(A):
    filas, columnas = A.shape
    for i in range(filas):
        suma = 0
        for j in range(columnas):
            suma += abs(A[i][j])
        if suma - abs(A[i][i]) >= abs(A[i][i]):
            return False
    return True

def matrizCirculante(v):
    tamaño = len(v)
    res = np.zeros((tamaño, tamaño))
    for i in range(tamaño):
        for j in range(tamaño):
            res[i][(j + i) % tamaño] = v[j]
    
    return res


def matrizVandermonde(v):
    tamaño = v.shape[0]
    res = np.zeros((tamaño,tamaño))
    for i in range(tamaño):
        for j in range(tamaño):
            res[i][j] = v[j]**i
    return res


def producto_mat(A,B):
    filas_a = A.shape[0]
    columnas_b = B.shape[1]
    res = np.zeros((filas_a, columnas_b))
    Bt = traspuesta(B)

    for i in range(filas_a):
        for j in range(columnas_b):
            res[i][j] = productoEscalar(A[i],Bt[j])
    return res



def potencia(A, n):
    if n == 1:
        return A
    return producto_mat(A,potencia(A,n-1))

def fibonacci(n):
    if n == 0:
        return 0
    A = np.array([[1,1],[1,0]])
    res = producto_mat(potencia(A,n), np.array([[1],[0]]))
    return res[1][0]



def numero_aureo(n):
    A = np.array([[1,1],[1,0]])
    res = producto_mat(potencia(A,n+1), np.array([[1],[0]]))
    return res[0][0]/res[1][0]


def graficar():
    numeros = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    res = np.zeros((1, 10))
    
    for i in range(10):
        res[0][i] = numero_aureo(i)
    
    plt.plot(numeros, res[0])
    plt.title("Aproximación al número áureo")
    plt.xlabel("n (n-ésimo Fibonacci)")
    plt.ylabel("Número áureo aproximado")
    plt.show()

# Ejecutar la función de graficado
#graficar()


def matrizFibonacci(n):
    res = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            res[i][j] = fibonacci(i+j)
    return res


def matrizHilbert(n):
    res = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            res[i][j] = 1/(1+j+1)
    return res


def raicesEcuaciones(numeroDeEcuacion):
    # Medir tiempo de inicio
    start_time = time.time()

    valores = np.linspace(-1, 1, 100000)
    
    f = [lambda x: x**5 - x**4 + x**3 - x**2 + x - 1, lambda y: y**2 + 3, lambda z: z**10 - 2]
    
    # Graficar
    plt.plot(valores, f[numeroDeEcuacion](valores), '*')
    plt.show()

    # Medir tiempo de ejecución
    end_time = time.time()
    tiempo_ejecucion = end_time - start_time
    
    # Calcular el tamaño en memoria de las variables
    memoria_valores = sys.getsizeof(valores)
    memoria_resultado = sys.getsizeof(f[numeroDeEcuacion](valores))
    
    print(f"Tiempo de ejecución: {tiempo_ejecucion:.6f} segundos")
    print(f"Memoria utilizada por 'valores': {memoria_valores / 1024:.6f} KB")
    print(f"Memoria utilizada por el resultado de f(valores): {memoria_resultado / 1024:.6f} KB")


def row_echelon(A):
    """ Return Row Echelon Form of matrix A """

    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    maximoPivot = 0
    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    for i in range(len(A)):
        if A[i,0] != 0 and abs(A[i,0]) > abs(A[maximoPivot,0]): 
            maximoPivot = i

    if maximoPivot > 0:
        ith_row = A[maximoPivot].copy()
        A[maximoPivot] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]

    # we perform REF on matrix from second row, from second column
    B = row_echelon(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

A = np.array([[4, 7, 3, 8],
              [8, 3, 8, 7],
              [2, 9, 5, 3]], dtype='float')

#print(row_echelon(A))


####TESTS
def tests():
    # Test 1: esCuadrada
    A1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A2 = np.array([[1, 2], [3, 4], [5, 6]])
    assert esCuadrada(A1) == True
    assert esCuadrada(A2) == False

    # Test 2: triangSup
    A1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A2 = np.array([[5, 2], [3, 8]])
    assert np.array_equal(triangSup(A1), np.array([[0, 2, 3], [0, 0, 6], [0, 0, 0]]))
    assert np.array_equal(triangSup(A2), np.array([[0, 2], [0, 0]]))

    # Test 3: triangInf
    assert np.array_equal(triangInf(A1), np.array([[0, 0, 0], [4, 0, 0], [7, 8, 0]]))
    assert np.array_equal(triangInf(A2), np.array([[0, 0], [3, 0]]))

    # Test 4: diagonal
    assert np.array_equal(diagonal(A1), np.array([[1, 0, 0], [0, 5, 0], [0, 0, 9]]))
    assert np.array_equal(diagonal(A2), np.array([[5, 0], [0, 8]]))

    # Test 5: traza
    assert traza(A1) == 15
    assert traza(A2) == 13

    # Test 6: traspuesta
    assert np.array_equal(traspuesta(A1), np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]))
    assert np.array_equal(traspuesta(A2), np.array([[5, 3], [2, 8]]))

    # Test 7: sonIguales
    assert sonIguales(A1, A1) == True
    assert sonIguales(A1, A2) == False

    # Test 8: esSimetrica
    assert esSimetrica(matrizSimetrica) == True
    assert esSimetrica(A1) == False

    # Test 9: productoEscalarnp.array([1, 2, 3])
    b = np.array([4, 5, 6])
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert productoEscalar(a, b) == 32
    c = np.array([7, 8, 9])
    assert productoEscalar(a, c) == 50

    # Test 10: calcularAx
    x = np.array([1, 0, 1])
    assert np.array_equal(calcularAx(A1, x), np.array([4, 10, 16]))
    assert np.array_equal(calcularAx(A2, x[:2]), np.array([5, 3]))

    # Test 11: intercambiarFilas
    A_copy = A1.copy()
    intercambiarFilas(A_copy, 0, 2)
    assert np.array_equal(A_copy, np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]]))

    # Test 12: sumaVectores
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert np.array_equal(sumaVectores(a, b), np.array([5, 7, 9]))

    # Test 13: vectorPorEscalar
    assert np.array_equal(vectorPorEscalar(a, 3), np.array([3, 6, 9]))
    assert np.array_equal(vectorPorEscalar(a, -1), np.array([-1, -2, -3]))

    # Test 14: sumarFilaMultiplo
    A_copy = A1.copy()
    sumarFilaMultiplo(A_copy, 1, 0, 2)
    assert np.array_equal(A_copy, np.array([[1, 2, 3], [6, 9, 12], [7, 8, 9]]))

    # Test 15: esDiagonalmenteDominante
    A1 = np.array([[3, 1, 1], [1, 4, 1], [1, 1, 5]])
    A2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert esDiagonalmenteDominante(A1) == True
    assert esDiagonalmenteDominante(A2) == False

    # Test 16: matrizCirculante
    v = np.array([1, 2, 3])
    assert np.array_equal(matrizCirculante(v), np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]]))
    v2 = np.array([4, 5])
    assert np.array_equal(matrizCirculante(v2), np.array([[4, 5], [5, 4]]))



    # Test 17: matrizVandermonde
    v = np.array([1, 2, 3])
    assert np.array_equal(matrizVandermonde(v), np.array([[1, 1, 1], [1, 2, 3], [1, 4, 9]]))


    # Test 18: producto_mat
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    assert np.array_equal(producto_mat(A, B), np.array([[19, 22], [43, 50]]))


    print("Pasaron todos")


#tests()




