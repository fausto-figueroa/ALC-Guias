import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import librerias as lib


def norma(x,p):
    res = 0
    if p == 'inf':
        for elem in x:
            if res < abs(elem):
                res = abs(elem)
    else:
        for elem in x:
            res += abs(elem)**p
        res = res**(1/p)
    return res

def normaliza(X, p):
    res = X.copy()
    for i in range(len(X)):
        vector = X[i]
        res[i] = vector/norma(vector,p)
    
    return res

def normaMatMC(A,q,p,Np):
    n = A.shape[0]
    normalizados = normaliza(np.random.randn(Np,n),p)
    normaMaxima = 0
    vectorMaximo = normalizados[0]
    for i in range(Np):
        normaActual = norma(lib.calcularAx(A,normalizados[i]),q) #norma de Ax con x normalizado 
        if normaActual > normaMaxima:
            normaMaxima = normaActual
            vectorMaximo = normalizados[i]
    return normaMaxima,vectorMaximo


def normaExacta(A,p=[1, 'inf']):
    if p == 'inf':
        suma_filas = []
        for i in range (A.shape[0]):
            suma_filas.append(norma(A[i],1))
        norma_inf = max(suma_filas)
        return norma_inf
    elif p == 1:
        suma_columnas = []
        for i in range(A.shape[1]):
            suma_columnas.append(norma(A[:,i],1))
        norma_1 = max(suma_columnas)
        return norma_1
    else:
        return None
    

def condMC(A,p,Np):
    A_inv = np.linalg.inv(A)
    return normaMatMC(A,p,p,Np)[0] * normaMatMC(A_inv,p,p,Np)[0]


def condExacto(A,p):
    return normaExacta(A,p) * normaExacta(np.linalg.inv(A),p)



# Tests norma
assert(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
assert(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
assert(norma(np.random.rand(10),2)<=np.sqrt(10))
assert(norma(np.random.rand(10),2)>=0)

# Tests normaliza
for x in normaliza([np.array([1]*k) for k in range(1,11)],2):
    assert(np.allclose(norma(x,2),1))
for x in normaliza([np.array([1]*k) for k in range(2,11)],1):
    assert(not np.allclose(norma(x,2),1) )
for x in normaliza([np.random.rand(k) for k in range(1,11)],'inf'):
    assert( np.allclose(norma(x,'inf'),1) )

# Tests normaExacta

assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(normaExacta(np.random.random((10,10)),1)<=10)
assert(normaExacta(np.random.random((4,4)),'inf')<=4)

# Test normaMC

nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

A = np.array([[1,2],[3,4]])
nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
assert(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 


# Test condMC

A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

A = np.array([[3,2],[4,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))



# Test condExacta

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,1)
normaA_ = normaExacta(A_,1)
condA = condExacto(A,1)
assert(np.allclose(normaA*normaA_,condA))

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,'inf')
normaA_ = normaExacta(A_,'inf')
condA = condExacto(A,'inf')
assert(np.allclose(normaA*normaA_,condA))

#print("Pasaron todos los tests")