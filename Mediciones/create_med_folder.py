import os
import re
import shutil
import yaml

from datetime import datetime

fps = 250

folder_comment = 'Mod de fase'
MEDN = 'auto'

today_date = datetime.today()
info = {
        'ACCELEROMETER': {'n_points': None, 'used': False},
        'CAMERA': {'fps': fps,
                   'n_deformed_images': 3072,
                   'n_gray_images': 100,
                   'n_reference_images': 100,
                   'resolution': [1024, 1024],
                   'shutter_speed': 1/fps
                   },
        'DATE': today_date.strftime('%m-%d-%y'),
        'FUNCTION_GENERATOR': {'hz': 20,
                               'vpp': 18,
                               'freq_modulation': 2},
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

    # Creo la carpeta MEDN y las accelerometer, deformed, gray y reference
    os.mkdir(MED_folder_name)
    os.mkdir(MED_folder_name + '/HDF5')

    # Escribo el archivo info
    with open(f'{MED_folder_name}/info.yaml', 'w') as f:
        yaml.dump(info, f, default_flow_style=False, default_style=None)

new_med_folder(folder_comment=folder_comment, MEDN=MEDN)
