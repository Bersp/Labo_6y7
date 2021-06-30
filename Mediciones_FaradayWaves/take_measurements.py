import os
import re
import shutil
import yaml

from datetime import datetime

fps = 250
acelerometer_n_points = None

folder_comment = ''
MEDN = 'auto'

today_date = datetime.today()
info = {
        'ACELEROMETER': {'n_points': acelerometer_n_points, 'used': bool(acelerometer_n_points)},
        'CAMERA': {'fps': fps,
                   'n_images_deformed': 3072,
                   'n_images_gray': 200,
                   'n_images_reference': 200,
                   'resolution': [1024, 1024],
                   'shutter_speed': 1/fps
                   },
        'DATE': today_date.strftime('%m-%d-%y'),
        'FUNCTION_GENERATOR': {'hz': 20,
                               'vpp': 11,
                               'freq_modulation': None},
        'CALIBRATION': {'d': 21.0, # distancia entre el proyector y la cámara
                        'L': 103.0}, # altura de la camara al disco
        'PROJECTOR': {'image_to_proyect': 'Sin01.BMP'},
        'NOTES': ''
        }


def new_med_folder(folder_comment=None, MEDN='auto'):

    if MEDN == 'auto':
        # Veo cuál es la última medición y tomo el nuevo n como n + 1
        MEDN = re.findall('MED([0-9]+) -', ' '.join(os.listdir('.')))
        MEDN = max(map(int, MEDN)) + 1
    elif isinstance(MEDN, int):
       repeted_folder = [x for x in os.listdir('.')[:-1] if f'MED{MEDN}' in x]
       if repeted_folder:
           shutil.rmtree(repeted_folder[0])
    else:
        raise Exception("MEDN tiene que ser un entero o 'auto'")
        return

    MED_folder_name = f'MED{MEDN}'
    if folder_comment:
        MED_folder_name = f'{MED_folder_name} - {folder_comment}'
    MED_folder_name = f'{MED_folder_name} - {today_date.strftime("%m%d")}'

    # Creo la carpeta MEDN y las acelerometer, deformed, gray y reference
    os.mkdir(MED_folder_name)
    # NOTE: No creo estas carpetas para saber cuáles ya fueron pasadas
    #  os.mkdir(f'{MED_folder_name}/deformed')
    #  os.mkdir(f'{MED_folder_name}/gray')
    #  os.mkdir(f'{MED_folder_name}/reference')
    #  os.mkdir(f'{MED_folder_name}/white')
    if acelerometer_n_points:
        os.mkdir(f'{MED_folder_name}/acelerometer')

    # Escribo el archivo info
    with open(f'{MED_folder_name}/info.yaml', 'w') as f:
        yaml.dump(info, f, default_flow_style=False, default_style=None)

    # Escribo el archivo del acelerómetro
    if acelerometer_n_points:
        script_argv = f'"{MED_folder_name}/acelerometer/acceleration.csv" {acelerometer_n_points}'
        os.system(f'python ../Acelerometro/Arduino/serial_export.py {script_argv}')


new_med_folder(folder_comment=folder_comment, MEDN=MEDN)
