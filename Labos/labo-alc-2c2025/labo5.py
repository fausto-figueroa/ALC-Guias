import numpy as np
import librerias as lib
import labo3 as l3

def signo(x):
    return -1 if x < 0 else 1

def sumaMatrices(A, B):
    if A.shape != B.shape:
        print("ADVERTENCIA! Intentando sumar matrices de dimensiones diferentes.")
        return None
    res = np.zeros(A.shape)
    for i in range(A.shape[0]):
        res[i] = lib.sumaVectores(A[i], B[i])
    return res


def QR_con_GS(A,tol=1e-12,retornanops=False):
    n,m = A.shape
    if (n != m): return None, None, 0.
    Q = np.zeros((n,n))
    R = np.zeros((n,n))
    

    R[0][0]  = l3.norma(A[0:n,0], 2)
    if R[0][0] < tol: return None
    Q[0:n,0] = A[0:n,0] / R[0][0] 
    for j in range(1, n):
        q = A[0:n,j]
        for k in range(0, j):
            t=(Q[0:n,k])
            R[k][j] = lib.productoEscalar(t, q) #ver de trasponer despues de haber hecho reshape
            q = q - R[k][j] * Q[0:n,k]
            R[j][j] = l3.norma(q,2)
            if R[j][j] < tol: return None
            Q[0:n,j] = q / R[j][j]

    if retornanops:
        return Q,R,0
    else: 
        return Q, R


def QR_con_HH(A,tol=1e-12):
    m, n =A.shape
    R=A
    Q = np.eye(m)
    Hm = np.zeros((m,m))
    for k in range (n):
        x = R[k:m,k]
        a = (-1)*signo(x[0])*l3.norma(x,2)
        Id=np.eye(m-k)
        u = (lib.sumaVectores(x,-a*Id[0]))
        u = u.reshape(len(u),1) #Esto es para que en python sea en verdad un vector de mx1
        norma_u=l3.norma(u, 2)
        if (norma_u>tol):
            u=u/norma_u
            Hk = sumaMatrices(Id ,lib.producto_mat(-2*u,lib.traspuesta(u)))
            
            Hm[0:k,0:k] = np.eye(k)
            Hm[k:m,k:m] = Hk
            Hm[0:k,k:m] = np.zeros((k,m-k))
            Hm[k:m,0:k] = np.zeros((m-k,k))

            R = lib.producto_mat(Hm, R)
            Q = lib.producto_mat(Q, lib.traspuesta(Hm)) 
    return Q, R


def calculaQR(A, metodo='RH', tol=1e-12):
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol)
    else:
        print("El metodo \"", metodo ,"\"no es valido para calcular la descomposicion QR.")
        return None


# Tests

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)



# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

# --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)

print("Pasaron todos los tests")
