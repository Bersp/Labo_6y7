Workflow para las experiencias y el tratamiento de datos
========================================================

Calibracion de la camara
------------------------

#. Tomar imagenes de calibracion para la camara.
#. Calibrar la camara (obtener los coeficientes de distorsion radial).

Captura de imagenes de franjas 
------------------------------

#. Tomar imagenes ``grises`` (varias para promediar despues).
#. Tomar imagenes ``referencias`` (varias para promediar despues).
#. Tomar imagenes ``deformadas``.


Tratamiento de imagenes de franjas (FTP + Gerchberg)
----------------------------------------------------

#. Generar una imagen promedio de las ``grises``, llamada ``gris``. 
#. Generar una imagen promedio de las ``referencias``, llamada ``referencia``.
#. Calibrar la camara.
#. Desdistorsionar la ``gris``.
#. Desdistorsionar la ``referencia``.
#. Usar la imagen ``gris`` para determinar:
    - el centro del disco,
    - el radio del disco, y 
    - la posicion del rectangulo.
#. Generar una mascara dada por:
    - un valor de 1, en el interior del disco y del rectangulo,
    - un valor de 0, en toda otra posicion.
#. Recortar la ``gris`` y la ``referencia`` dejando en cada una solamente el
   disco y el rectangulo (y un poco de margen para que Gerchberg pueda
   periodizarlas!).
#. Extender por fuera de la mascara la ``gris`` usando el algoritmo de Gerchberg.
#. Extender por fuera de la mascara la ``referencia`` usando el algoritmo de Gerchberg.
#. Para cada imagen del set  ``deformadas``:
    #. Desdistorsionar la imagen.
    #. Recortar la imagen dejando el rectangulo y el disco (y un poco de
           margen, idem).
    #. Extender la imagen usando el algoritmo de Gerchberg.
    #. Procesar la imagen mediante FTP, generando un mapa de diferencia de fase `plegado`. 
    #. Desplegar el mapa de fase obtenido usando un algoritmo de `unwrap bidimensional` y suministrandole la mascara generada previamente, de forma que solo despliegue la fase dentro de esa region.
    #. Si correspondiese, ejecutar un `unwrapping temporal` respecto a la imagen `deformada` anterior.
    #. Separacion de la imagen en dos:
        - una cuadrada que contenga al disco inscripto, y 
        - una rectangular dentro del rectangulo de referencia 
    #. A las del disco restarles el :math:`<h_{rect}(x,y,t)>_{(x,y)}` 
    #. Almacenar:
        - :math:`<h_{rect}(x,y,t)>_{(x,y)}` 
        - :math:`h_{disk-wo-rect}(x,y,t)`





Almacenamiento masivo de datos crudos y de resultados de procesamiento 
----------------------------------------------------------------------

Cada serie de mediciones corresponde a:
    - aproximadamente 3000 imagenes a una resolucion de 1024 x 1024 pixeles, con una profundidad de color de 10 bits.
    - una senal temporal de datos (proveniente del acelerometro).
    - un numero fijo de datos asociados al setup de FTP y la excitacion provista al recipiente.

Esta cantidad de datos `crudos` requiere del empleo de una estrategia de
almacenamiento especializada. Nosotros hemos optado por realizar todo almacenamiento (y lectura) de datos empleando archivos tipo ``hdf5``, mediante la libreria/wrapper ``h5py``.


Organizacion de datos para el procesamiento automatizado por FTP
----------------------------------------------------------------

Cada serie de mediciones **crudas** deberia estar organizada de la siguiente manera:

::
    
    serie_2018_06_10_001/
        calibration_images/ 
            cal_im_0001.tif
            cal_im_0002.tif
            ...
            cal_im_WXYZ.tif
        ftp_images/
            gray/
            references/
            deformed/
        parameters/
            capture_parameters.yaml
        accelerometer_signal/
            ax_ay_az_t.csv 


El archivo ``capture_parameters.yaml`` es una lista estructurada en YAML, como la siguiente:

.. include:: capture_parameters.yaml
   :literal:

Como output del procesamiento, deberiamos tener:

::

    ftp_processing_2018_06_10_001/
        calibration_results/
            calibration_parameters.???
        height_fields/
            disk-wo-rectangle/
            rectangle/
        parameters/
            capture_parameters.yaml
            processing_parameters.yaml

Como output del analisis fourier para el calculo de espectros, deberiamos
tener:

::

    fourier_analysis_2018_06_10_001/
        fft2D/
        fft3D/
        comments/


Como output de la separacion entre moduladora y high-frequency, deberiamos
tener:

::

    hilbert_analysis_2018_06_10_001/
        envelope/
        high-frequency/
        comments/


Development stage
-----------------

Upon modification of the source code, run

::

    pipreqs modulation_instability/

to create an updated ``requirements.txt`` file for the module.
