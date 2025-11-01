import numpy as np
import librerias as lib
import labo3 as l3


def random_vector(tamaño):
    v = np.random.rand(tamaño)
    norma_v = l3.norma(v,2)
    if norma_v == 0:
        return random_vector(tamaño)
    else:
        return lib.vectorPorEscalar(v, 1/norma_v) #v/norma_v

def f(A,v):
    w_prima = lib.calcularAx(A,v)
    norma_w_prima = l3.norma(w_prima,2)
    if norma_w_prima == 0:
        return np.zeros(A.shape[0])
    else:
        return w_prima/norma_w_prima
    
def metpot2k(A,tol=1e-15,K=1000):
    tamaño = A.shape[0] #A es cuadrada
    v = random_vector(tamaño)
    v_moño = f(A,f(A,v))
    e = lib.productoEscalar(v_moño,v)
    k = 0
    while abs(e-1) > tol and k < K:
        v = v_moño
        v_moño = f(A, f(A,v))
        e = lib.productoEscalar(v_moño,v)
        k = k +1
    temp = lib.calcularAx(A,v_moño)
    lambda_ = lib.productoEscalar(v_moño, temp)
    return (v_moño,lambda_,k)    

def pad_diag(A, elem):
    n = A.shape[0]
    P = np.zeros((n+1,n+1))

    P[0,         0] = elem
    P[1:n+1, 1:n+1] = A
    P[1:n+1,     0] = np.zeros(n)
    P[0    , 1:n+1] = np.zeros((1,n))
    
    return P

def diagRH(A,tol=1e-15,K=1000):
    if not lib.esSimetrica(A):
        print("No es simetrica A=", A)
        return None
    
    n = A.shape[0]
    v1,l1,_ = metpot2k(A,tol,K)

    # u = (e1-v1)
    v1 *= -1
    v1[0] += 1
    u = v1

    Hv1   = np.eye(n) - 2 * (
                lib.producto_mat(
                    u.reshape(n, 1), u.reshape(1, n)
                ) / (l3.norma(u,2)**2)
            )
    
    M = lib.producto_mat(
            Hv1, lib.producto_mat(
                A, lib.traspuesta(Hv1)
            )
    )
    
    if n == 2:
        S = Hv1
        D = M
    else:
        B = M
        A_ = B[1:n,1:n]
        S_, D_ = diagRH(A_, tol, K)
        D = pad_diag(D_, l1)
        S = lib.producto_mat(Hv1, pad_diag(S_, 1))

    return S, D



# Tests

# Tests metpot2k

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
              ]).T

# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    v,l,_ = metpot2k(A,1e-15,1e5)
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95


#Test con HH
exitos = 0
for i in range(100):
    v = np.random.rand(9)
    #v = np.abs(v)
    #v = (-1) * v
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

    A = H@D@H.T
    v,l,_ = metpot2k(A, 1e-15, 1e5)
    #max_eigen = abs(D[0][0])
    if abs(l - D[0,0]) < 1e-8:         
        exitos +=1
assert exitos > 95



# Tests diagRH
D = np.diag([1,0.5,0.25])
S = np.vstack([
    np.array([1,-1,1])/np.sqrt(3),
    np.array([1,1,0])/np.sqrt(2),
    np.array([1,-1,-2])/np.sqrt(6)
              ]).T

A = S@D@S.T
SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
assert np.allclose(D,DRH)
assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    A = np.random.random((5,5))
    A = 0.5*(A+A.T)
    S,D = diagRH(A,tol=1e-15,K=1e5)
    ARH = S@D@S.T
    e = l3.normaExacta(ARH-A,p='inf')
    if e < 1e-5: 
        exitos += 1
assert exitos >= 95


