import matplotlib.pyplot as plt
import numpy as np

import serial

def main():

    # Settings
    serial_port = '/dev/ttyACM0'
    baud_rate = 115200
    ser = serial.Serial(serial_port, baud_rate)

    START_TOKEN = 'START'
    buffer_len = 5_000

    output_file = 'data.csv'

    # Espero a que descargue la info cargada de la última realización
    wait_start_token(ser, START_TOKEN)

    f = open(output_file, 'w')

    # Guardo los primeros 1000 datos antes de empezar a eliminar
    counter = 0
    while counter < buffer_len:
        line = read_serial_line(ser)
        print(line)
        f.write(line)

        counter += 1
    del counter

    while True:
        # NOTE: Falta buscar cómo eliminar la primera línea de forma razonable
        line = read_serial_line(ser)
        print(line)
        f.write(line)

def wait_start_token(ser, START_TOKEN):
    line = read_serial_line(ser)
    while START_TOKEN not in line:
        line = read_serial_line(ser)
        print('DATO EN BUFFER:', line)
    line = read_serial_line(ser)

def read_serial_line(ser):
    try:
        line = ser.readline()
        line = line.decode("utf-8")
    except UnicodeDecodeError: # solo puede tirar este error cuando comienza
        line = START_TOKEN
    return line

if __name__ == '__main__':
    main()
