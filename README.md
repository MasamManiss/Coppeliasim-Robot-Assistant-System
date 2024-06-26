# Coppeliasim-Robot-Assistant-System
# Voice Controlled Robot and Object Manipulation

This project allows you to control a robot and manipulate objects in a simulated environment using voice commands. The robot navigates through the environment, picks up colored spheres, and stores or moves them to different locations as instructed.


## User Installation Instructions

Follow these steps to set up and run the project on your local machine.

### Prerequisites

Make sure you have the following software and dependencies installed:

- Python (version 3.x)
- CoppeliaSim
- vosk
- pyaudio
- numpy
- opencv-python
- pathfinding (AStarFinder)
- spacy

### Step-by-Step Installation


1. Install the necessary Python packages:
   ```bash
   pip install coppeliasim zmq pyaudio numpy opencv-python pathfinding spacy
   ```
2. Download this [Python file](https://github.com/MasamManiss/Robot-Assistant-System/blob/main/Final.py)   
3. Download and install [CoppeliaSim](https://www.coppeliarobotics.com/)
   
4. Download this [CoppeliaSim Scene](https://github.com/MasamManiss/Robot-Assistant-System/blob/main/Final.ttt)
   
5. Open the scene and run the python file simultaneously
6. How to Operate
   Once the program is run. You are prompt to press 'spacebar' to record your command in 4 seconds. Below are some of the commands you can execute.

   | Navigation Command | Object Retrieval Command |
   |--------------------|--------------------------|
   | Go to room 1       | Take red ball            |
   | Move to room 2     | Pass me the blue ball    |
   | Head to room 3     | Store this               |
