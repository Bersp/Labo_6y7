import matplotlib.pyplot as plt
import numpy as np

import serial

def main():
    global START_TOKEN

    # Settings
    serial_port = '/dev/ttyACM0'
    baud_rate = 115200
    ser = serial.Serial(serial_port, baud_rate)

    START_TOKEN = 'START'
    buffer_len = 500


    output_file = 'data.csv'

    # Espero a que descargue la info cargada de la última realización
    wait_start_token(ser, START_TOKEN)

    with open(output_file, 'w') as f:
        f.write('')

    data = ''

    # Guardo los primeros 1000 datos antes de empezar a eliminar
    #  counter = 0
    #  while counter < buffer_len:
        #  line = read_serial_line(ser)
        #  data += line
        #  with open(output_file, 'w') as f:
            #  f.writelines(data)

        #  counter += 1
    #  del counter
    while True:
        line = read_serial_line(ser)
        #  data = data.split('\n', 1)[1] + line
        data = data + line
        with open(output_file, 'a') as f:
            f.write(line)

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

if __name__ == '__main__':
    main()
