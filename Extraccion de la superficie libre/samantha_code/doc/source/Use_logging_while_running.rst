Use logging while running
-------------------------

In order to use logging messages while running commands from any 
module in this package, you should do the following in ipython command line.

.. code-block:: python
   :linenos:

    import logging
    logging.getLogger('modulation_instability').setLevel(logging.INFO)
    logging.getLogger('bulk_processing').setLevel(logging.INFO)
    from modulation_instability import bulk_processing
    bulk_processing.process_datafile_by_ftp(data_series_file='/Volumes/Seagate Backup Plus Drive/Modulation_Instability/2018-07-17-0001-RAW.hdf5', parameter_file='/Volumes/Seagate Backup Plus Drive/Modulation_Instability/ftp_processing_parameters.yaml')


Afterwards you call the functions in the usual way. Logging for that
module will be displayed with level ``INFO``. It can be changed to
any other level in a similar way.


