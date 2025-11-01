import numpy as np
import librerias as lib

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


def calculaLU(A):
    L = None
    U = None
    numeroDeOperaciones = 0

    fila,col = A.shape

    Ac = A.copy()

    for i in range(col-1):
        pivot = Ac[i][i]
        if pivot == 0:
            return L,U,numeroDeOperaciones
        for j in range(i+1, fila):
            c = Ac[j][i]/pivot
            numeroDeOperaciones += 1
            Ac[j,i:col] = Ac[j, i:col] - c * Ac[i,i:col]
            numeroDeOperaciones += 2*(col - i)
            Ac[j][i] = c
    
    L = triangInf(Ac) + np.eye(fila)

    U = triangSup(Ac) + diagonal(Ac)

    return L,U,numeroDeOperaciones

def res_tri(L,b,inferior=True):
    b_len = b.shape[0]
    res = np.zeros(b_len)
    if inferior == True:
        res[0] = b[0]/L[0][0]
        for i in range(1,b_len):
            Lc = L[i,0:i+1]
            res[i] = b[i]
            for j in range(Lc.shape[0]-1):
                res[i] -= res[j] * Lc[j]
            res[i] /= L[i,i]
    else:
        res[b_len-1] = b[b_len-1]/L[b_len-1][b_len-1]
        for i in range(b_len-2, -1, -1):
            Lc = L[i,i:b_len]
            res[i] = b[i]
            for j in range(Lc.shape[0]-1):
                res[i] -= res[b_len-1-j] * Lc[Lc.shape[0]-1-j]
            res[i] /= L[i,i]
    return res

def inversa(A):

    L, U, _ = calculaLU(A)
    tamanio = U.shape[0]
    # Si es singular devolvemos None
    for i in range(tamanio):
        if U[i][i] == 0: return None

    I = np.eye(tamanio)

    U_inv = np.zeros((tamanio,tamanio))
    L_inv = np.zeros((tamanio,tamanio))
    for i in range(tamanio):
        U_inv[0:tamanio,i] = res_tri(U,I[i],False)
        L_inv[0:tamanio,i] = res_tri(L,I[i],True)

    return lib.producto_mat(U_inv, L_inv)
    
def calculaLDV(A):

    nops = "cambiame"
    L, U, _ = calculaLU(A)

    tamanio = A.shape[0]
    D = np.zeros((tamanio, tamanio))
    for i in range(tamanio):
        if U[i][i] == 0: return None, None, None, None
        D[i][i] = U[i][i]


    D_inv = np.zeros((tamanio, tamanio))
    for i in range(tamanio):
        D_inv[i][i] = 1/D[i][i]
    V = lib.producto_mat(D_inv,U)  # U es diagonal asi que U = U^t
    
    
    return L, D, V, nops

def esSDP(A,atol=1e-8):
    L, D, V, nops = calculaLDV(A)

    if L is None: return False

    for i in range(D.shape[0]):
        if D[i][i] <= 0: return False
    return lib.esSimetrica(A)



# Tests LU

L0 = np.array([[1,0,0],[0,1,0],[1,1,1]])
U0 = np.array([[10,1,0],[0,2,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(np.allclose(L,L0))
assert(np.allclose(U,U0))


L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
U0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(not np.allclose(L,L0))
assert(not np.allclose(U,U0))
assert(np.allclose(L,L0,atol=1e-3))
assert(np.allclose(U,U0,atol=1e-3))
#assert(nops == 13)

L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
U0 = np.array([[1,1,1],[0,0,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(L is None)
assert(U is None)
#assert(nops == 0)


## Tests res_tri

A = np.array([[1,0,0],[1,1,0],[1,1,1]])
b = np.array([1,1,1])
assert(np.allclose(res_tri(A,b),np.array([1,0,0])))
b = np.array([0,1,0])
assert(np.allclose(res_tri(A,b),np.array([0,1,-1])))
b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b),np.array([-1,2,-2])))
b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([-1,1,-1])))

A = np.array([[3,2,1],[0,2,1],[0,0,1]])
b = np.array([3,2,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1/3,1/2,1])))

A = np.array([[1,-1,1],[0,1,-1],[0,0,1]])
b = np.array([1,0,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1,1,1])))        
        

# Test inversa

ntest = 10
iter = 0
while iter < ntest:
    A = np.random.random((4,4))
    A_ = inversa(A)
    if not A_ is None:
        assert(np.allclose(np.linalg.inv(A),A_))
        iter += 1

# Matriz singular devería devolver None
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
assert(inversa(A) is None)

# Test LDV:

L0 = np.array([[1,0,0],[1,1.,0],[1,1,1]])
D0 = np.diag([1,2,3])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
A =  L0 @ D0  @ V0
L,D,V,nops = calculaLDV(A)
assert(np.allclose(L,L0))
assert(np.allclose(D,D0))
assert(np.allclose(V,V0))

L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
D0 = np.diag([3,2,1])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1.001]])
A =  L0 @ D0  @ V0
L,D,V,nops = calculaLDV(A)
assert(np.allclose(L,L0,1e-3))
assert(np.allclose(D,D0,1e-3))
assert(np.allclose(V,V0,1e-3))

# Tests SDP

L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
D0 = np.diag([1,1,1])
A = L0 @ D0 @ L0.T
assert(esSDP(A))

D0 = np.diag([1,-1,1])
A = L0 @ D0 @ L0.T
assert(not esSDP(A))

D0 = np.diag([1,1,1e-16])
A = L0 @ D0 @ L0.T
assert(not esSDP(A))

L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
D0 = np.diag([1,1,1])
V0 = np.array([[1,0,0],[1,1,0],[1,1+1e-10,1]]).T
A = L0 @ D0 @ V0
assert(not esSDP(A))

print("Pasaron todos los tests")