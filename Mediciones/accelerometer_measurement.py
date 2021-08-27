import os
import re

import yaml

MEDN = 15
accelerometer_n_points = 5_000

MED_folder_name, = [x for x in os.listdir('.') if f'MED{MEDN} - ' in x]

yaml_filename = f'{MED_folder_name}/info.yaml'

with open(f'{MED_folder_name}/info.yaml', 'r') as f:
    yaml_dict = yaml.safe_load(f)

yaml_dict['ACCELEROMETER']['n_points'] = accelerometer_n_points
yaml_dict['ACCELEROMETER']['used'] = True

with open(f'{MED_folder_name}/info.yaml', 'w') as f:
    yaml.dump(yaml_dict, f, default_flow_style=False, default_style=None)

os.makedirs(f'{MED_folder_name}/accelerometer', exist_ok=True)
script_argv = f'"{MED_folder_name}/accelerometer/acceleration.csv" {accelerometer_n_points}'

os.system(f'python ../Acelerometro/Arduino/accelerometer_export.py {script_argv}')
