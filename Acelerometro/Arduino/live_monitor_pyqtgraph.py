from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time
import serial

app = QtGui.QApplication([])

p = pg.plot()
p.setWindowTitle('live plot from serial')
curve = p.plot()

#  raw=serial.Serial('ACM0', 115200)

with open('data/data_baud115200_bw1600_20Hz_4vpp.txt') as f:
    lines = f.readlines()
def get_lines():
    for l in lines:
        yield l

lin_gen = get_lines()

data = [next(get_lines()).split(', ')[3]]
def update():
    global curve, data
    #  line = raw.readline()
    line = next(lin_gen).split(',')
    data.append(float(line[3]))
    xdata = np.array(data, dtype='float64')
    curve.setData(xdata)
    app.processEvents()

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

