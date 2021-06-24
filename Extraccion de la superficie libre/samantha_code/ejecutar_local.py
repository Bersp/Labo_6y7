import numpy as np
import h5py
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
from unwrap import unwrap
import telegram
import logging
import sys
sys.path.insert(0, '/home/samantha/Dropbox/2020_Kucher_Samantha/fiteo_GL/')
from modulation_instability import bulk_processing
from modulation_instability import input_output
from modulation_instability import analysis
logging.getLogger('modulation_instability').setLevel(logging.INFO)
import time
import telegram

path_mediciones = '/media/samantha/Seagate Backup Plus Drive/Samantha/mediciones-2019-10-16/tarde/'
destination_path = '/media/samantha/Seagate Backup Plus Drive/Samantha/mediciones-2019-10-16/tarde/a120_f20_fmod0_apagando/'
path_calibraciones = '/media/samantha/Seagate Backup Plus Drive/Samantha/Calibraciones/'
def_folder = 'def_a120_f20_fmod0_apagando/'
filename = input_output.read_parameter_file(path_mediciones+def_folder+'capture_parameters.yaml')['DATA_SERIES_NAME']

print(def_folder)
bot = telegram.Bot(token=bot_id)

#Crear archivo hdf5
logging.getLogger('input_output').setLevel(logging.INFO)
start_time_hdf5 = time.time()
input_output.create_original_dataset(
            destination_path=destination_path,
            parameter_file=path_mediciones+def_folder+'capture_parameters.yaml',
            cam_calib_path=path_calibraciones+'calibracion_camara/',
            acc_calib_path=path_calibraciones+'arduinoDUE/',
            ftp_path=path_mediciones,
            deformed_folder=path_mediciones+def_folder,
            accel_file='')
tiempo_hdf5 = time.time()-start_time_hdf5

#Procesado FTP
logging.getLogger('bulk_processing').setLevel(logging.INFO)
start_time_ftp = time.time()
bulk_processing.process_datafile_by_ftp(destination_path = destination_path, data_series_file= destination_path+filename+'-RAW.hdf5', parameter_file= path_mediciones+'processing_parameters.yaml')
tiempo_ftp = time.time()-start_time_ftp

bot.send_message(chat_id=my_id, text='Termino el procesado FTP en '+str(round(tiempo_ftp/60,2))+' minutos. Pasando a coordenadas polares')
#bot.send_message(chat_id=my_id, text="Ya terminó el procesado FTP. Pasando a coordenadas polares.")
#bot.send_message(chat_id=pablo_id, text="Ya terminó el procesado FTP. Pasando a coordenadas polares.")

logging.getLogger('analysis').setLevel(logging.INFO)
start_time_coord_polares = time.time()
analysis.save_height_annulus_polar_coordinates(file=destination_path+filename+'-annulus-PRO.hdf5')
tiempo_coord_polares = time.time()-start_time_coord_polares



'''
#Transformada de Fourier
#logging.getLogger('analysis').setLevel(logging.INFO)
#start_time_fourier_disco = time.time()
#analysis.save_fourier_transform(data_file= destination_path+filename+'-disk-PRO.hdf5', analysis_file= destination_path+filename+'-disk-FOURIER.hdf5')
#tiempo_fourier_disco = time.time()-start_time_fourier_disco

logging.getLogger('analysis').setLevel(logging.INFO)
start_time_fourier_anillo = time.time()
analysis.save_annulus_fourier_transform(data_file=destination_path+filename+'-annulus-PRO.hdf5', analysis_file=destination_path+filename+'-annulus-FOURIER.hdf5', create_new_file=True)
tiempo_fourier_anillo = time.time()-start_time_fourier_anillo

#Descomposicion EMD
#logging.getLogger('analysis').setLevel(logging.INFO)
#start_time_emd_disco = time.time()
#analysis.save_envelope_carrier(data_file=destination_path+filename+'-disk-PRO.hdf5', analysis_file=destination_path+filename+'-EMD.hdf5')
#tiempo_emd_disco = time.time()-start_time_emd_disco

logging.getLogger('analysis').setLevel(logging.INFO)
start_time_emd_anillo = time.time()
analysis.save_envelope_carrier_annulus(data_file=destination_path+filename+'-annulus-PRO.hdf5', analysis_file=destination_path+filename+'-annulus-EMD.hdf5', create_new_file=True)
tiempo_emd_anillo = time.time()-start_time_emd_anillo


'''
mensaje = str(filename) +'\n' \
+'Crear hdf5: '+str(round(tiempo_hdf5/60,2))+' minutos'+'\n' \
+'Procesado FTP: '+str(round(tiempo_ftp/60,2))+' minutos'+'\n' \
+'Coordenadas polares: '+str(round(tiempo_coord_polares/60,2))+' minutos'+'\n' \
+'Tiempo total: '+str(round(round(tiempo_hdf5/60,2)+round(tiempo_ftp/60,2)+round(tiempo_coord_polares/60,2))) +' minutos'

# mensaje = str(filename) +'\n' \
# +'Procesado FTP: '+str(round(tiempo_ftp/60,2))+' minutos'+'\n' \
# +'Coordenadas polares: '+str(round(tiempo_coord_polares/60,2))+' minutos'+'\n' \
# +'Tiempo total: '+str(round(round(tiempo_ftp/60,2)+round(tiempo_coord_polares/60,2))) +' minutos'

print(mensaje)

f = h5py.File(destination_path+filename+'-annulus-PRO.hdf5','r')
p = f['height_fields/annulus_polar']
fps = 250
R1, R2 = (205/2)/10, (215/2)/10 #en cm
Rprom = (R1+R2)/2
xfinal = 2*np.pi*Rprom

plt.figure()
plt.imshow(unwrap(p), extent=[0,xfinal, 0, p.shape[1]/fps], aspect='auto',cmap='RdYlBu')
plt.clim((-1,1))
plt.colorbar()
plt.xlabel('longitud de arco (cm)')
plt.ylabel('tiempo (s)')
plt.title(filename)
plt.tight_layout()
plt.savefig(path_mediciones+'graficos/'+filename+'.png')

bot.send_photo(chat_id=my_id, photo=open(path_mediciones+'graficos/'+filename+'.png', 'rb'), caption=mensaje)
#bot.send_photo(chat_id=pablo_id, photo=open(path_mediciones+'graficos/'+filename+'.png', 'rb'), caption=mensaje)
