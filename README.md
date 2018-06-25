# EEG_Event_Classification
To run this project....

0. git clone this project

1. Download the files hosted here: https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_events/v1.0.0/

2. Order the train/test files such that they resemble the following in the same directory as the rest of the files:
  -v1.0.0
   --edf
    ---test
      ---000
      ---001
      ---...
    ---train 
      ---0000017
      ---0000018
      --- ...
   
   
3. While in this folder, run ```python3 EEG_DataCleaning.py```. Run it on a computer with a LOT of RAM, I personally had to use a an ec2 instance to run that command in less than an hour. 

4. After that, run ```python3 EEG_Classification_Final.py```. You definitely need a powerful computer for this, do not run w/o a GPU, a single epoch will take hours. 

5. Run EEG_Accuracy_Analysis.py from iPython to inspect the produced confusion matrices individually. 


  
