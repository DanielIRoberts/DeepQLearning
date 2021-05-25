General Info---------------------------------------------------------------------------------------------------

Name	- Daniel Roberts	
	- Daoqun Yang			
IDE	- Visual Studio Code
Python Version - 3.7.2

Included Packages----------------------------------------------------------------------------------------------

import numpy as np
import gym
import keras

Included Files-------------------------------------------------------------------------------------------------

Model.py
	- Class ModelBuild
		1). Initialize all zeros for random action
		2). Store action, current space, reward, newSpace and Endspace
		3). Randomize indicies and getting sample
		4). Build Deep Q learning model with two hidden layer and using relu activation function
		5). Input, reshaping and output layers.
	- Class Learner
		1). Initialize gamma, epsilon, samplesize, epsilonDec, epsilonEnd and Numact
		2). Using a random number to compare epsilon so that it can choose action
		3). Read data, get target model and predict next state using Q equation
		4). Update epsilon value each time learn
		5). Assign different points depend on different situation
	- Main Fuction
		1). Gain user input
		2). Build game environment
		3). Print game info and iteration info
		4). Save data into txt file
		5). Save model to .h5 file

Example Input----------------------------------------------------------------------------------------------------

Please input model name: Finalproject.h5
Please input file name: Finalout.txt

Description-----------------------------------------------------------------------------------------------------

This program utilizes the gym package to teach a deep reinforcement learning agent how to play the classic game 
taxi. The log files in the various versions contain models and changes made to the agents to help the agent play
the game more effectively. The main problem the agent initially ran into was a scarce reward problem. This was 
overcome by giving it points not just for succesfully picking up and dropping off a rider, but also being 
within a block of the rider. To counteract the agent abusing the points from being near the rider, I removed 
points if the agent repeated actions too often. I have also attached the final report submitted with this 
project.

----------------------------------------------------------------------------------------------------------------
