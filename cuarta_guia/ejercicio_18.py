import numpy as np

def producto_interno(v, w):
    if len(w) != len(v):
        print("dimensiones erroneas")
        return
    res = 0
    for i in range(len(w)):
        res += w[i]*v[i]
    return res

def Ax(A,x):
    if A.shape[1] != len(x):
        print("dimensiones erroneas")
        return
    res = np.zeros(A.shape[0])

    for i in range(A.shape[1]):
        res[i] = producto_interno(A[i], x)
    return res

def fun_lineal(v):
    res = 0
    for coord in v:
        res += coord
    return res

def raleigh(x,v):
    return producto_interno(x,v)

def metodo_de_la_potencia(A, f, K=100, tol= 1e-15):
    n = A.shape[0]
    if A.shape[1] != n:
        print("Debe ser una Matriz Cuadrada")
        return
    v = np.random.rand(n) #np.array([0,1,1]) este tiene a1 = 0
    if v.all() == 0:
        print("mala suerte, el random te dio todo 0 jiji")
        return
    
    w = Ax(A,v)
    autovalor = f(v,w) / f(v,v)
    k = 0

    while k < K and v.all() > tol:
        v = w
        w = Ax(A,v)
        autovalor = f(v,w) / f(v,v)
        k +=1
    return autovalor

A = np.array([[-6,9,3],[0,8,-2],[0,-1,7]])
print("el mio:",metodo_de_la_potencia(A, raleigh))
print("el otro", (np.linalg.eig(A)))