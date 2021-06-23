import matplotlib.pyplot as plt
import numpy as np
import sys

import serial

def wait_start_token(ser, START_TOKEN):
    counter = 0
    line = read_serial_line(ser)
    while START_TOKEN not in line:
        line = read_serial_line(ser)
        print('DATO EN BUFFER:', line)
        counter += 1
    line = read_serial_line(ser)

def read_serial_line(ser):
    global START_TOKEN
    try:
        line = ser.readline()
        line = line.decode("utf-8")
    except UnicodeDecodeError: # solo puede tirar este error cuando comienza
        line = START_TOKEN
    return line


argv = sys.argv
if len(argv) > 2:
    output_file = argv[1]
    N = int(argv[2])
    print_meds = False
else:
    N = 1e4
    output_file = 'data.csv'
    print_meds = True

# Settings
#  serial_port = '/dev/ttyACM0'
serial_port = '/dev/pts/9'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate)

START_TOKEN = 'START'


# Espero a que descargue la info cargada de la última realización
wait_start_token(ser, START_TOKEN)

# Borro el archivo anterior
with open(output_file, 'w') as f:
    f.write('')

# Guardo N datos
for _ in range(N):
    line = read_serial_line(ser)
    if print_meds: print(line)
    with open(output_file, 'a') as f:
        f.write(line)
