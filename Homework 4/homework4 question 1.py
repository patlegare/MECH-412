import control as ct
import numpy as np
import matplotlib.pyplot as plt

#Transfer function a)

s=ct.tf("s")
G_1= 50/(s*(1+s/10)*(1+(s/1000)))
print(G_1)

#Transfer function b)
G_2=(1000*((s+10)**2))/(s**2+14*s+100)
print(G_2)

#plot
w = np.logspace(-2, 5, 1000)
ct.bode_plot([G_1,G_2],omega=w,dB=True, Hz=False, deg=True,margins=False)
plt.legend(["G1(s)", "G2(s)"])
plt.show()