Usage example
=============

In this section we describe a usage example of the MIMo code. We assume that
the user already has already collected the *minimal raw data files* described 
in the next section.

Minimal raw data files
----------------------

The *minimal raw data files* necessary for running this example consists on
the following **five** individual sets:

1. Camera calibration data: 
    a series of camera calibration images, stored in ``path_to_calibration_camera``, 

2. Accelerometer calibration data: 
    a series of accelerometer calibration arrays, stored in ``path_to_calibration_accelerometer``, 

3. FTP data:
    a series of gray, reference and deformed images, stored in 
    ``path_to_ftp_images`` and organized acording to:

::

    path_to_ftp_images/
        gray/
            gray_image_0001.tif
            gray_image_0002.tif
            ...
            gray_image_IIII.tif
        reference/
            reference_image_0001.tif
            reference_image_0002.tif
            ...
            reference_image_JJJJ.tif
        deformed/
            deformed_image_0001.tif
            deformed_image_0002.tif
            ...
            deformed_image_KKKK.tif

4. Raw accelerometer data:
    a file containing a time-series reading of a 3-axis accelerometer, named 
    ``raw_accelerometer_readings``.

5. Parameter file:
    a ``capture_parameters.yaml`` file which gathers the detailed configuration
    of the experimental setup employed for capturing each individual data set.
    This ``capture_parameters.yaml`` file should look like the following:

    .. include:: capture_parameters.yaml
        :literal:


Generating a monolithic raw dataset
-----------------------------------

Up until this stage, the *minimal raw data set* is just a bunch of files 
scattered on the user's hard drive. Therefore, the main objective of this
first stage of processing is to **collect** all this info together in **one 
monolithic file** in which all info is held and accessed as a unit.

This is done in the following way. Let's begin by launching an interactive
python session and using MIMo to create this file. 

.. code-block:: python
   :linenos:

   from modulation_instability.input_output import create_original_dataset   
   create_original_dataset(parameter_file = 'capture_parameters.yaml', \ 
       cam_calib_path = 'path_to_calibration_images', \
       acc_calib_path = 'path_to_accelerometer_calibration_arrays' \
       ftp_path = 'path_to_ftp_images', accel_file = 'raw_accelerometer_readings')


*Note that the function* ``create_original_dataset`` *takes named arguments, so that they can
be assigned in any order.*

After running these commands, a HDF5 file is created and named as stated by 
the value of the variable ``DATA_SERIES_NAME`` (according to the 
``capture_parameters.yaml`` file) with a ``-RAW`` suffix added to denote 
the fact that this dataset file holds only raw data.  
In this example case, the file will be 
named ``2018-07-17-0001-RAW.hdf5``. This ``hdf5`` file is organized internally 
as follows:

::

    2018-07-17-0001-RAW.hdf5/
        calibration_camera
        calibration_accelerometer
        ftp_images/
            gray
            reference
            deformed
        raw_accelerometer_readings


In this scheme, ``calibration_camera``, ``calibration_accelerometer`` and ``raw_accelerometer_readings`` 
are HDF5 datasets, whereas ``ftp_images`` is a HDF5 group holding three datasets:
``gray``, ``reference`` and ``deformed``. The parameter values detailed in
the ``capture_parameters.yaml`` file are now stored as attributes associated
to each of the corresponding datasets in the newly created HDF5 file. 

In detail, each of the contents of the ``-RAW`` file is:

- ``calibration_camera`` is a 3D dataset, each plane [:,:,z] holds one of the 
  calibration images.
- ``calibration_accelerometer`` is a 3D dataset of size :math:`4 \times N_t
  \times N_p`, where :math:`N_p` is the number of postures employed in the 
  calibration and :math:`N_t` is the number of time-points in each posture
  measurement.
- ``ftp_images`` is a group which holds three datasets related to FTP
  measurements:
    - ``gray`` is a 3D dataset, each plane [:,:,z] holds one of the gray
      images,
    - ``reference`` is a 3D dataset, each plane [:,:,z] holds one of the
      reference images, and
    - ``deformed`` is a 3D dataset, each plane [:,:,z] holds one of the
      deformed images.
- ``raw_accelerometer_readings`` is a 4-column-by-N-lines 2D dataset, where
  each column corresponds to time, x-acceleration, y-acceleration and
  z-acceleration **raw** data, i.e., as read by Arduino without conversion to
  physical units.


How to generate a monolithic raw dataset **reusing** calibration results from another dataset
---------------------------------------------------------------------------------------------

Due to the fact that recalibration is only necessary whenever the setup changes
(this is true, in our case, for the accelerometer and the camera), it might be
useful to be able to **reuse** calibration data from another (already
processed) dataset. This can be done in the following way.

Let's suppose that we would like to reuse the accelerometer calibration results
which are part of a HDF5 ``-PRO`` file (a datafile with results from a previous
processing of a ``-RAW`` datafile), named ``2018-07-10-0001-PRO``. In that case, we do:

.. code-block:: python
   :linenos:

   from modulation_instability.input_output import create_original_dataset   
   create_original_dataset(parameter_file = 'capture_parameters.yaml', \ 
       cam_calib_path = 'path_to_calibration_images', \
       acc_calib_path='2018-07-10-0001-PRO.hdf5' \
       ftp_path = 'path_to_ftp_images', accel_file = 'accelerometer_readings')


By passing ``2018-07-10-0001-PRO`` to ``acc_calib_path`` (instead of a folder with
accelerometer calibration arrays), the function ``create_original_dataset``
understands that it should take the calibration results for the accelerometer
from that particular file. 

The ``-RAW`` file created in this way presents the following internal organization:

::

    2018-07-17-0001-RAW.hdf5/
        calibration_camera
        res_calibration_accelerometer
        ftp_images/
            gray
            reference
            deformed
        raw_accelerometer_readings




Bulk-processing of the raw dataset
----------------------------------

What do we mean by data bulk-processing 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``-RAW`` data file holds all raw data together, the only difference between
this and the raw files being that now all our data is *together* and
*organized* in one place: a HDF5 file. In order to work with this data, we need
to process it. Concretely, this processing stage involves the following
operations on raw data:

- **from the accelerometer calibration data**, calibrate it to obtain the three
  sensibility and the three offset coefficients :math:`(S_x, S_y, S_z, O_x,
  O_y, O_z)` and their uncertainties,
- **from the camera calibration images**, calibrate the camera to obtain a 
  camera matrix :math:`MATRIX` containing (amongst others) the radial and
  tangential distortion parameters, and 
- **from the fringe images**, calculate the height or phase difference maps.
  In turn, this part of the processing involves masking, fringe extrapolation
  and FTP processing.


How to bulk-process a data file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


For the data processing of this newly created monolithic raw dataset, the user needs
to prescribe the value of certain parameters employed in the processing. These
are established in a YAML file named ``ftp_processing_parameters.yaml``, whose
contents are as follows:

.. include:: ftp_processing_parameters.yaml
    :literal:

With this settings file and the monolithic raw dataset created on the previous
stage, it is straightforward to process the dataset by FTP. The following lines 
of code describe this operation with MIMo.

.. code-block:: python
   :linenos:

    from modulation_instability.bulk_processing import process_datafile_by_ftp 
    process_datafile_by_ftp(data_series_file='/Volumes/Seagate Backup Plus Drive/Modulation_Instability/20
        18-07-17-0001-RAW.hdf5', parameter_file='/Volumes/Seagate Backup Plus Drive/Modulation_Instability/ftp_processing_parameters.yaml', logging_level='logging.INFO')

The command ``process_datafile_by_ftp`` creates a new HDF5 file, named
``DATA_SERIES_NAME`` with the added suffix ``-PRO``, indicating that it holds 
**processed** datasets. In this example case, the created file name is
``2018-07-17-0001-PRO.hdf5``. Its internal structure is as follows:

::

    2018-07-17-0001-PRO.hdf5/
        calibration_results/
        height_fields/
            disk_wo_rectangle
            rectangle

Again, the parameter values prescribed in the
``ftp_processing_parameters.yaml`` file are now stored as attributes associated
to each of the corresponding datasets in the resulting HDF5 file.



Individual processing of parts of the raw dataset
-------------------------------------------------

While exploring data, it might be useful to have a way to perform a particular
processing on just a part of the dataset. For example, the user may be willing
to quickly check whether the calibration images for the camera lead to a good
estimation of the distortion parameters, **before** launching the
bulk-processing of all data (including the often lengthy calculation of height
fields via FTP).

In the following paragraphs we include some examples on how to perform such
operations, assuming a ``-RAW`` HDF5 file already exists. 


One complete run
----------------

.. code-block:: python
   :linenos:

    $ ipython

    import logging

    from modulation_instability import input_output

    logging.getLogger('modulation_instability').setLevel(logging.INFO)

    logging.getLogger('input_output').setLevel(logging.INFO)

    input_output.create_original_dataset(parameter_file='/Volumes/Seagate Backup Plus Drive/Modulation_Insta
    bility/capture_parameters.yaml', cam_calib_path='/Volumes/Seagate Backup Plus Drive/Modulation_Instabili
    ty/Subset_of_229/Calibracion', acc_calib_path='/Users/pablo/', ftp_path='/Volumes/Seagate Backup Plus Dr
    ive/Modulation_Instability/Subset_of_229', accel_file='', destination_path='/Volumes/Seagate Backup Plus
    Drive/Modulation_Instability/')

    INFO:modulation_instability.input_output:Creando el dataset de la camara...
    INFO:modulation_instability.input_output:Creando dataset de grises
    INFO:modulation_instability.input_output:Creando dataset de referencia
    INFO:modulation_instability.input_output:Creando dataset deformadas
    INFO:modulation_instability.input_output:Listo.


.. code-block:: python
   :linenos:

    logging.getLogger('bulk_processing').setLevel(logging.INFO)

    from modulation_instability import bulk_processing

    bulk_processing.process_datafile_by_ftp(data_series_file='/Volumes/Seagate Backup Plus Drive/Modulation_
    Instability/2018-07-17-0001-RAW.hdf5', parameter_file='/Volumes/Seagate Backup Plus Drive/Modulation_Ins
    tability/ftp_processing_parameters.yaml', destination_path='/Volumes/Seagate Backup Plus Drive/Modulatio
    n_Instability/')

    INFO:modulation_instability.bulk_processing:Reading ftp_processing_parameters
    INFO:modulation_instability.bulk_processing:Creating destination hdf5 file
    INFO:modulation_instability.bulk_processing:Calibrating camera
    INFO:modulation_instability.bulk_processing:Generate averages gray and reference, and generate mask
    INFO:modulation_instability.bulk_processing:FTP processing






Performing just the camera calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's suppose that we only want to perform the camera calibration, and 
check the results obtained in just this stage of the data processing.
The following code does the trick:

.. code-block:: python
   :linenos:

   import h5py
   hf = h5py.File('2018-07-17-0001-RAW.hdf5','r')
   dset = hf['calibration_camera']
   from modulation_instability.calibration import calibrate_camera
   mtx, roi, mapx, mapy = calibrate_camera(dset, [8, 6], 1)


The code works as follows. As we will need to access a HDF5 ``-RAW`` data file,
we import the module ``h5py`` [line 1]. Line 2 opens the file for reading. 
Line 3 grabs the relevant dataset, ``calibration_camera``, and puts it into
``dset``. Line 4 imports the ``calibrate_camera`` function, and the last line
performs the actual calibration (using a 8x6 checkerboard) and 
sending its output to the variables ``mtx``, ``roi``, ``mapx``, and ``mapy``.


Performing just the accelerometer calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Performing fringe extrapolation on just one fringe image 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

   import h5py
   hf = h5py.File('2018-07-17-0001-RAW.hdf5','r')
   # Grab one gray image and generate mask
   dsetg = hf['ftp_images/gray']
   # mask = 
   dsetd = hf['ftp_images/deformed']
   # Grab one deformed image
   img = dset[:,:,0]
   # Fill using Gerchberg extrapolation, providing fringe image & mask
   # fe_img =  
   


