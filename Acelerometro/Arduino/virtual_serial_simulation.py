import serial
import time
import numpy as np

def f(x): return np.round(np.sin(x), 4)

ser =  serial.Serial('/dev/pts/4', 115200)
#  ser.write(b'START')

x = 0
while True:
    x += 2*np.pi/30
    ser.write(f'{x},{x},{x},{f(x)}\n'.encode('utf-8'))
    time.sleep(0.001)
