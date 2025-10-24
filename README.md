# DJI-Tello-Lane-Following-with-Python-OpenCV-CNN-and-Behavioral-Clonning
This is a Repository For all the .py files we needed to create the autonomous lane following program for the DJI Tello Edu

### A few words about the project.

This is a replicated project from a masters subject from the University of West Attica in Greece. The original project was dealing the creation of a line following program using deep learning techniques.

Inspired by this, we decided to replicate the original project and make the drone to stay in between the 2 black lines on the floor
The project however has a few weaknesses that may not allow the drone to complete the designated floor circuit. On the contrary this is a first attempt of a student that aspires to finish his intergraded masters diploma and make a thesis about the autonomous indoor flight of drone platforms.

Link to the original Project: https://aidl.uniwa.gr/students-projects/following-a-track-with-a-dji-tello-drone-using-deep-learning/

MsC program link: https://aidl.uniwa.gr/

### Instructions

1.	First make sure that the DJI Tello Edu drone, is able to connect with you computer via the wifi gate.
Note: If not able to connect, please refer to the website of the DJI for troubleshooting.
Click Here to see the Manual

2.	Install Pycharm

3.	Install Python 3.11 on your Computer.


4.	Download all the files from the GitHub Folder: tello_line_following
Take that file and put it on Users->user->PycharmProjects

5.	Go to pycharm and press file->open->PycharmProjects->tello_line_following
6.	From the terminal of pycharm check witch version of Python you have: 
python --version
7.	Go to the terminal and write the following command: ``` pip install -r requirments.txt ```
8.	Alternatively open powercell or command line and with the cd command reach the path you have the file. You should see something like this C:\Users\user\PycharmProjects\tello_line_following> 
Then write the pip install -r requirments.txt

10.	Then open the file collect_data.py. By opening it and hitting run, the stream shoud open and the drone should start flying.
By pressing the keyboard keys:
•	A: Stears to the Left
•	D: Stears to the Right
•	S: Stops the drone and allow it to hover.
•	Up Arrow: Increases the Altitude
•	Down Arrow: Decreases the Altitude
9a.  If you want to change the parameters. Open the file with the name config.py. Then change the parameter needed to the desired goal.
When the drone starts to fly imedietlty begins to take samples of the circuit. The samples are all stored in the directory :
C:\Users\user\PycharmProjects\tello_line_following\data\session_YYYY-MM-DD_HH-MM-Seconds
Every session_YYYY-MM-DD_HH-MM-Seconds indicates the time, day, month, year the flight took place.
Every new run is a new session.
11.	Open the train.py file. On line with the code: 
SESSION_PATH = Path("data\session")
Inside the :“”, put the path of the session you want the data  to use in order to train the CNN network.
12.	Run the train.py file and let it run. It might take a while…
Note: By default we set the epochs at 150 because we hat large numbers of samples (more than 12.000). 
For 6.000 to 9.000 samples ->100 to 130 epochs 
For 9.000 to 13.000 samples ->130 to 150 epochs
For 13.000> samples -> 150 to 180 epochs
These numbers are not binding and depending the case you will have to adjust depending the results of the confusion matrix to avoid overfitting and underfitting.
13.	Open the Iinference.py file and run. This file will take the latest policy that was trained in the previous step. To avoid conflict we keep only the latest policy. After every training the previous one is overwritten.
14.	Everything is done!!!
