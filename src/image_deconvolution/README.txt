Sample yaml files:
tra_events.yaml: This is used for parsing the .tra file when creating an unbinned event hdf5 file, and it is also used for parsing the unbinned event hdf5 file in order to bin the data. Contains editable binning parameters, and the specified tra file name. 
housekeeping_data_io.yaml: This is used within DataIO to specify the detector geometry and mimrec configuration files, which can be found in COSItools/massmodel-comptonsphere.

Testing files:
TestSource.520.inc1.id1.tra.gz: The tra file used for testing, but any of the tra files in /volumes/selene/users/evangela/simulationScript or /volumes/selene/users/andreas/simulationScript/Output should do.
image_deconvolution_test_main.ipynb: The main Jupyter notebook script for testing the image deconvolution algorithm.

The below 3 classes have been edited for our purposes. So far, everything else has been unchanged.
data_loader.py: Edited load_from_filepath() and set_event_from_filepath() to read in the event data from a .tra file and output the unbinned data in an hdf5 file as an intermediate step. This data is then binned and outputted as an hdf5 file. Also, I think you could completely skip the step of reading in the data as a tra file, and edit it to read in just the .pkl files!
UnBinnedData.py: Edited to exclude spacecraft pointing information. Assumes fixed galactic coordinates, i.e. with detector always pointing directly "up." In this sense, the local coordinates coincide with the galactic coordinates.  
BinnedData.py: Edited for our local coordinate system (technically a fixed galactic frame), i.e. no correction for spacecraft frame.