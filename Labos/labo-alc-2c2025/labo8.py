import numpy as np
import math 
import librerias as lib
import labo6 as l6
import labo3 as l3

def sigma(D, k, tol=1e-15):
    sigma = np.zeros((k, k))
    for i in range(k):
        if D[i,i] > tol:
            sigma[i,i] = math.sqrt(D[i,i]) 

    return sigma

def ui(sigma, A, vi):  # (1/sigma_i) * A * v_i

    Avi = lib.calcularAx(A, vi)
    return (Avi / sigma)

def svd_reducida(A,k="max",tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """
    m, n = A.shape
    AAt = lib.producto_mat(A, lib.traspuesta(A))
    AVALS, AVECS = l6.diagRH(AAt, tol) # A * At simetrica 

    if k == "max":
        k = min(m, n) # chequear

    SIGMA = sigma(AVALS, k, tol)

    Vk = np.zeros((k, n))             # Ya esta traspuesta!
    for i in range(k):
        Vk[i] = AVECS[:,i] / l3.norma(AVECS[:,i], 2)  # O dividir por norma? esta raro normaliza
    
    Uk = np.zeros((m, k))
    for i in range(k):
        Uk[:,i] = ui(SIGMA[i,i], A, Vk[i])  # u_i = (1/sigma_i) * A * v_i

    return Uk, SIGMA, Vk




    
# Matrices al azar
def genera_matriz_para_test(m,n=2,tam_nucleo=0):
    if tam_nucleo == 0:
        A = np.random.random((m,n))
    else:
        A = np.random.random((m,tam_nucleo))
        A = np.hstack([A,A])
    return(A)

def test_svd_reducida_mn(A,tol=1e-15):
    m,n = A.shape
    hU,hS,hV = svd_reducida(A,tol=tol)
    nU,nS,nVT = np.linalg.svd(A)
    r = len(hS)+1
    assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol), 'Revisar calculo de hat U en ' + str((m,n))
    assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol), 'Revisar calculo de hat V en ' + str((m,n))
    assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
    assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))

for m in [2,5,10,20]:
    for n in [2,5,10,20]:
        for _ in range(10):
            A = genera_matriz_para_test(m,n)
            test_svd_reducida_mn(A)


# Matrices con nucleo

m = 12
for tam_nucleo in [2,4,6]:
    for _ in range(10):
        A = genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
        test_svd_reducida_mn(A)

# Tamaños de las reducidas
A = np.random.random((8,6))
for k in [1,3,5]:
    hU,hS,hV = svd_reducida(A,k=k)
    assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
    assert hU.shape[1] == k, 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[1] == k, 'Dimensiones de hV incorrectas(caso a)'
    assert len(hS) == k, 'Tamaño de hS incorrecto'
