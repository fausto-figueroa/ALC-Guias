import numpy as np 

v1 = np.array([10, 5,-7,1,0,0,0])
print(v1)
v2 = np.array([10,10,10,10,10,10,7.100])
print(v2)

print("\naca estan las matrices\n")

A = np.array([[2,2],[1,3]])
print("A= \n",A)

B = np.array([[1,2],[4,5]])
print("B= \n", B)

print("v1[1]= ", v1[1], "\n")
print("A[0][0]=", A[0][0], "\n")

print("v1+v2 = ", v1 + v2)

print("2A =", 2*A)

print("A + B = ", A+B)