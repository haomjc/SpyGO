import numpy as np
from math import pi
from math import log, sqrt
import matplotlib.pyplot as plt

# Try to evaluate the solution of Bussinesq integral and find the trend w.r.t. to the distance 

def Szz(iprime, jprime, i, j, hx, hy, ni, G):
    """
    z direction compliance on point (i,j) for a uniform rectangular pressure distribution on (iprime,jprime)
    """
    deltai=iprime-i
    deltaj=jprime-j
    k=(deltai+0.5)*hx
    m=(deltaj+0.5)*hy
    l=(deltai-0.5)*hx
    n=(deltaj-0.5)*hy

    sk2n2 = np.sqrt(k**2+n**2)
    sl2m2 = np.sqrt(l**2+m**2)
    sk2m2 = np.sqrt(k**2+m**2)
    sl2n2 = np.sqrt(l**2+n**2)

    S = (1-ni)/(2*pi*G)
    return S*(k*np.log((m+sk2m2)/(n+sk2n2))\
              +l*np.log((n+sl2n2)/(m+sl2m2))\
                +m*np.log((k+sk2m2)/(l+sl2m2))\
                    +n*np.log((l+sl2n2)/(k+sk2n2)))

def main():
    ni = 0.3
    E = 206000
    G = E/2/(1+ni)
    t = np.linspace(-2, 2, 1000)
    X, Y = np.meshgrid(t, t)
    print(X)
    Kzz_matrix = Szz(0,0,t,t,0.1,0.1, ni, G)
    contourplot = plt.imshow(Szz(np.array([0]),np.array([0]),X,Y,np.array([0.1]),np.array([0.1]), ni, G), extent=[t.min(), t.max(), t.min(), t.max()])
    plt.colorbar(contourplot)
    plt.title('Colormap of Szz(x,y)')
    plt.show()

    # or show along the cross-section
    x = X[500, :]
    Kz = Szz(0, 0, 0, t, 0.1, 0.1, ni, G)
    plt.plot(x, Kz)
    plt.title('Compliance Szz')
    plt.show()

    print(Szz(0,0,1,1,0.1,0.1, ni, G))
    return

if __name__ == '__main__':
    main()