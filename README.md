# Temperatures_Interpolater
This repository contains the temperature data from Yang ZN, Dong YL, Xu WJ (2013) Fire tests on two-way concrete slabs in a full-scale multi-storey steel-framed building. Fire Saf J 58:38–48. https://doi.org/10.1016/j.firesaf.2013.01.023

This data is extracted from the figures in the paper using webplotdigitizer. 

- data_processing_modules: contains custom 'time_regularisation' Python module and corresponding tests.
- data_processing_notebooks: contains Jupyter notebooks that use the data grabbed from the figures and the time_regularisation module to produce OpenSees-compatible temperature files.
- digitised_image_data: digitised data and project package for the figures from the paper produced using webplotdigitiser. The .tar files can be imported into webplotdigitiser.
- processed_temp_files: contains the OpenSees-compatible temperature data files.
- images: contains the figures from the paper