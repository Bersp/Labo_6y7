import os
import shutil
import yaml


from datetime import datetime

fps = 250
acelerometer_n_points = 10

today_date = datetime.today()
info = {
        'ACELEROMETER': {'n_points': acelerometer_n_points, 'used': bool(acelerometer_n_points)},
        'CAMERA': {'fps': fps,
                   'n_images_deformed': 1, # TODO: Eliminar los 1
                   'n_images_gray': 1,
                   'n_images_reference': 1,
                   'resolution': [1024, 1024],
                   'shutter_speed': 1/fps
                   },
        'DATE': today_date.strftime('%d/%m/%y'),
        'FUNCTION_GENERATOR': {'hz': 20000,
                               'vpp': 9.5},
        'NOTES': ''
        }


def new_med_folder(folder_comment=None, MEDN='auto'):

    if MEDN == 'auto':
        # Veo cuál es la última medición y tomo n + 1
        MEDN = max([int(x.split('-', 1)[0].replace('MED', ''))
               for x in os.listdir('.')[:-1]]) + 1
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
    MED_folder_name = f'{MED_folder_name} - {today_date.strftime("%d%m")}'

    # Creo la carpeta MEDN y las acelerometer, deformed, gray y reference
    os.mkdir(MED_folder_name)
    os.mkdir(f'{MED_folder_name}/deformed')
    os.mkdir(f'{MED_folder_name}/gray')
    os.mkdir(f'{MED_folder_name}/reference')
    os.mkdir(f'{MED_folder_name}/acelerometer')

    # Escribo el archivo info
    with open(f'{MED_folder_name}/info.yaml', 'w') as f:
        yaml.dump(info, f, default_flow_style=False, default_style=None)

    # Escribo el archivo del acelerómetro
    if acelerometer_n_points:
        script_argv = f'"{MED_folder_name}/acelerometer/acceleration.csv" {acelerometer_n_points}'
        os.system(f'python ../Acelerometro/Arduino/serial_export.py {script_argv}')


print(new_med_folder(MEDN=0, folder_comment='Medicion de prueba'))
