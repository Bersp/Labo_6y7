Behind the scenes
-----------------

In this section we describe how the processing of a ``-RAW`` datafile is
performed.

Processing of fringe images
^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Calibration of the camera to obtain distortion parameters. 

#. Generation of one averaged gray image.

#. Generation of one averaged reference fringe image.

#. Undistort gray image. 

#. From gray image, obtain:

    - mask where the disk is (where fluid is held),
    - center and radius of disk,
    - mask where the vessel height reference is,
    - position of the vessel height reference center.

#. Undistort reference image.

#. Generation of (reference-gray) image. [This one is already undistorted,
   because it is created from two undistorted ones].

#. Extrapolate fringes of (reference-gray) image beyond the masks obtained
   precedently.

#. For each deformed image:

    #. Undistort deformed image.

    #. Calculate (deformed - gray). 

    #. Extrapolate fringes of (deformed-gray) image beyond the masks (of disk &
       rectangle).

    #. Calculate the unwrapped phase difference map between (deformed-gray) and
       (reference-gray).

    #. Time unwrap using a given point inside the real interferogram.

    #. Calculate mean phase difference over rectangular (reference) area.

    #. Substract, from phase difference map over the disk, the mean phase
       difference calculated previously.

    #. Calculate height field via the phase-to-height relation.

    #. Apply original circular mask to the calculated height field.

    #. Crop the height field to a square holding only the disk. For this we use
       the position of the center of the disk as well as its radius, as
       calculated earlier.

    #. This cropped height field is stored as one slice of a 3D dataset.


.. Calibration of the accelerometer
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



    
