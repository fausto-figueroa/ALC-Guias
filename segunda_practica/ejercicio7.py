import numpy as np

def ej(case):
    eps = np.finfo(float).eps
    print (eps)
    if(case == 1):    
        p = 1e34
        q = 1
        return p + q - p #aca me da mal, da 0.0
    elif(case == 2):
        p = 100
        q = 1e-15
        return (p+q)+q, ((p+q)+q+q), p + 2*q, p + 3*q #aca me da todo 100, o sea p
    elif(case ==3):
        return 0.1 + 0.2 == 0.3 #aca me da mal
    elif(case ==4):
        return 0.1 + 0.3 == 0.4 #aca me da bien
    elif(case ==5):
        return 1e-323 #aca me da bien
    elif(case ==6):
        return 1e-324 #aca me da 0.0
    elif(case ==7):
        return eps/2 #aca da bien
    elif(case ==8):
        return (1+eps/2)+eps/2 #aca da 1
    elif(case ==9):
        return 1+(eps/2 + eps/2) #aca da bien
    elif(case ==10):
        return ((1+eps/2)+eps/2) -1 #aca da 0
    elif(case ==11):
        return (1+(eps/2 + eps/2)) -1 #aca da bien
    elif(case ==12):
        list=[0]*25
        for i in range(1,26):
            list[i-1] = np.sin(10**i*np.pi)
        return list
    elif(case ==13):
        list=[0]*25
        for i in range(1,26):
            list[i-1] = np.sin(np.pi/2 + np.pi*10**i)
        return list
    

print(ej(10))