import numpy as np

def error(x,y):
    return abs(np.float64(x)-np.float64(y))

def error_relativo(x,y):
    return error(x,y)/abs(x)

def matricesIguales(A,B):
    eps = 1e-08

    filas,columnas = A.shape

    if A.shape != B.shape:
        return False

    for i in range(filas):
        for j in range(columnas):
            e = error(A[i,j],B[i,j])
            if e >= eps:
                return False

    return True

def sonIguales(x,y,atol=1e-08):
  return np.allclose(error(x,y),0,atol=atol)
assert(not sonIguales(1,1.1))
assert(sonIguales(1,1 + np.finfo('float64').eps))
assert(not sonIguales(1,1 + np.finfo('float32').eps))
assert(not sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
assert(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps,atol=1e-3))
assert(np.allclose(error_relativo(1,1.1),0.1))
assert(np.allclose(error_relativo(2,1),0.5))
assert(np.allclose(error_relativo(-1,-1),0))
assert(np.allclose(error_relativo(1,-1),2))

assert(matricesIguales(np.diag([1,1]),np.eye(2)))
assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
assert(not matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))
assert(not matricesIguales(np.array([[1 + np.finfo('float32').eps,2],[3,4]]),np.array([[1,2],[3,4]])))


print("Pasaron todos los tests")
