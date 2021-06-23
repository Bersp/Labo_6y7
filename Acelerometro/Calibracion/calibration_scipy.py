import numpy as np

def get_Vk():
    V = np.zeros(shape=(3, 16))
    for k in range(0, 16):
        t, x, y, z = np.loadtxt(f'data_calibration/calibration{k}.csv',
                                delimiter=',', skiprows=31, unpack=True)
        v_k = np.asarray([x.mean(), y.mean(), z.mean()])

        V[:,k] = v_k
    return V

vk = get_Vk()
v = np.matrix([0,0,1e2]).T
#  print(vk)

def get_E(S, O, V):
    e_k =
