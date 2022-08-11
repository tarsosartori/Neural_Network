# Neural_Network

## Table of contents
* [General infos](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Data Recovery](#data-recovery)

## General Info
 This program simulates rotation movements through a set of three gimbals connected, the inner, the middle, and the outer gimbal. Gimbals are like frames that can rotate independently. The inner gimbal may represent any object rotating in three dimensions, such as a car, a rocket, an airplane, a satellite, or even a hand. The composition of three independent rotations can reach any orientation in space. To achieve this, the program has been made using quaternions to represent the orientation and the quaternions kinematics to produce the movement. At the end of each simulation, the program shows the final quaternion, i.e., the orientation of the inner gimbal with respect to the inertial reference frame, as well the Euler angles associated using the convention 321. 
 ## Technologies
 The technologies used to build this program were:
 * Python 3.9
 * Numpy
 * Scipy
 * Matplotlib
 * Tkinter
 
 ## Setup
 To run this project you need to execute the python script named "program_gimbals.py". For this you need to have installed Python 3.9 or more recent versions. To check your python version open a linux terminal or the cmd for Windows and type:
 ```
 python3 -V
 ```
 To install pip:
 ```
 sudo apt install pip3
```
 And the libraries numpy, scipy, matplotlib:
 ```
 pip3 install matplotlib=3.4.3
 pip3 install numpy
 pip3 install scipy
 sudo apt-get install python3-tk 
 ```
 To run the program, go to the folder where is the script "program_gimbals.py" and execute:
 ```
 python3 program_gimbals.py
 ```
 You are going to see the two figures:
 
 
![Data](./images/window.PNG)


You can choose the parameters for each simulation and click on run. You can make as many as you want. Just a disclaimer, neither increase the sample time nor decrease the simulation time so much. If you do, the simulation can be harmed due to errors in the matrix exponentials it calculates. 


![Gimbals system](./images/gimbal.PNG)


The gimbals will rotate accordingly. In this figure, the outer gimbal is in blue, the middle is in green and the inner is in red. For each run, the program will show the final quaternion and the Euler angles for the inner gimbal in the terminal. 

## Data Recovery
To retrieve all the quaternions calculated during the simulation, you can copy the data stored in the variable quat1. This variable is a matrix with five columns. In the first column is the sample time of the simulation. The other four columns are the quaternion components according to the convention 0123, where the first component is the real part and the others are the imaginary part. The Euler angles can be recovered through the Euler321rad variable, which contains the angles for roll, pitch, and yaw in each column in radians.
