# ISA-for-AVs
**Identifying and Explaining Safety-critical Scenarios for Autonomous Vehicles via Key Features**

This repository contains the data and scripts to replicate the experimental results reported in the article "Identifying and Explaining Safety-critical Scenarios for Autonomous Vehicles via Key Features".
Directories Dataset1 and Dataset2 contain test scenarios and the features extracted from the test suites used in the paper. These directories also include the Python scripts to extract the features from the test scenarios. The details of the files included in these directories are given below:

1. Scenarios contains the test scenarios generated for testing the AVs.
2. metadata.csv contains the feature set extracted from the test scenarios. For test suite 1, another csv, complete-feature-set.csv is also provided. It contains a complete set of features listed in Table 1 of the article. 
3. feature-extraction.py contains the Python code for the extraction of features from the scenarios. This code is available for test suite 1 only. For test suite 2, the features are extracted using the SDC-Scissor tool available at https://github.com/christianbirchler-org/sdc-scissor.
4. Classification.py contains the code for model training and assessment.

