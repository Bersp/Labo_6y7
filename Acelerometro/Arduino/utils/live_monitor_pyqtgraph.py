from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time
import serial

app = QtGui.QApplication([])

buffer_len = 500

p = pg.plot()
#  p.setYRange(-512, 512, padding=0)
p.setWindowTitle('live plot from serial')
curve = p.plot()

#  raw=serial.Serial('ACM0', 115200)

#  with open('data.csv') as f:
    #  lines = f.readlines()
#  def get_lines():
    #  for l in lines:
        #  yield l

#  lin_gen = get_lines()

data = [0]
def update():
    global curve, data
    z = []
    #  while len(z) != buffer_len:
    t, x, y, z = np.loadtxt('data.csv', delimiter=',', unpack=True)
    curve.setData(t, z)
    app.processEvents()

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

