import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import librerias as lib


def rota(theta):
    rotacion = np.array([[np.cos(theta), -(np.sin(theta))],[np.sin(theta), np.cos(theta)]])
    return rotacion

def escala(s):
    n = len(s)
    res = np.eye(n)
    for i in range(n):
        res[i][i] = s[i]
    return res

def rota_y_escala(theta,s):
    return lib.producto_mat(rota(theta),escala(s))

def afin(theta, s, b):

    re = rota_y_escala(theta,s)
    res = np.array([
        [re[0][0],re[0][1],b[0]],
        [re[1][0],re[1][1],b[1]],
        [0,       0,       1   ]

    ])
    return res

def trans_afin(v,theta,s,b):
    aux = np.array([v[0],v[1],1])
    ta = lib.calcularAx(afin(theta,s,b), aux)
    return np.array([ta[0],ta[1]])


# Tests para rota
assert np.allclose(rota(0), np.eye(2))
assert np.allclose(rota(np.pi / 2), np.array([[0, -1], [1, 0]]))
assert np.allclose(rota(np.pi), np.array([[-1, 0], [0, -1]]))

# Tests para escala
assert np.allclose(escala([2, 3]), np.array([[2, 0], [0, 3]]))
assert np.allclose(escala([1, 1, 1]), np.eye(3))
assert np.allclose(escala([0.5, 0.25]), np.array([[0.5, 0], [0, 0.25]]))

# Tests para rota_y_escala
assert np.allclose(
    rota_y_escala(0, [2, 3]),
    np.array([[2, 0], [0, 3]])
)
assert np.allclose(
    rota_y_escala(np.pi / 2, [1, 1]),
    np.array([[0, -1], [1, 0]])
)
assert np.allclose(
    rota_y_escala(np.pi, [2, 2]),
    np.array([[-2, 0], [0, -2]])
)

# Tests para afin
assert np.allclose(
    afin(0, [1, 1], [1, 2]),
    np.array([
        [1, 0, 1],
        [0, 1, 2],
        [0, 0, 1]
    ])
)
assert np.allclose(
    afin(np.pi / 2, [1, 1], [0, 0]),
    np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])
)
assert np.allclose(
    afin(0, [2, 3], [1, 1]),
    np.array([
        [2, 0, 1],
        [0, 3, 1],
        [0, 0, 1]
    ])
)

# Tests para trans_afin
assert np.allclose(
    trans_afin(np.array([1, 0]), np.pi / 2, [1, 1], [0, 0]),
    np.array([0, 1])
)
assert np.allclose(
     trans_afin(np.array([1, 1]), 0, [2, 3], [0, 0]),
     np.array([2, 3])
 )

print("Pasaron todos los tests")

