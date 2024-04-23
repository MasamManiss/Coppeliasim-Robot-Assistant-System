from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement
import vosk
import pyaudio
import json
import keyboard
import time
import math
import cv2
import numpy as np
import spacy
import spacy.training.iob_utils as iob_utils
import random
from spacy.training.example import Example

recognizer = vosk.Model(r'C:\Users\lenovo\Desktop\Python\Vosk\vosk-model-en-us-0.42-gigaspeech')
#recognizer = vosk.Model(r'C:\Users\lenovo\Desktop\Python\Vosk\vosk-model-small-en-us-0.15')
# Create a recognizer object
rec = vosk.KaldiRecognizer(recognizer, 16000)

# Create an audio stream using PyAudio
audio = pyaudio.PyAudio()

# Define audio stream parameters
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)

# Set the recording duration (in seconds)
record_duration = 4 
print("Press 'Space' key to start record for 4 seconds")

# nlp
train = True
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
# Define a dictionary of synonyms
synonyms = {
    "go": ["go", "move", "head", "direct", "navigate"],
    "take": ["pick up", "give", "pick"],
    "store": ["keep, put"],# Add more synonyms as needed
}
#global variable
# Define range of color in HSV
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])
#
print("Program Started")
#setup api
client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

sim.startSimulation()
#
#initialize
manipulator = sim.getObject('/UR3')
obj1 = sim.getObject('/Sphere1')
obj2 = sim.getObject('/Sphere2')
obj3 = sim.getObject('/Sphere3')
tip = sim.getObject('/UR3/BaxterVacuumCup/tip')
sensor = sim.getObject('/Vision_sensor')
front = sim.getObject('/PioneerP3DX/front')
robot = sim.getObject('/PioneerP3DX')
alpha = sim.getObject('/alpha')
beta = sim.getObject('/beta')
gamma = sim.getObject('/gamma')
position_front = sim.getObjectPosition(front)
position_robot = sim.getObjectPosition(robot)
position_alpha = sim.getObjectPosition(alpha)
position_beta = sim.getObjectPosition(beta)
position_gamma = sim.getObjectPosition(gamma)
left_motor_handle = sim.getObject('/PioneerP3DX/leftMotor')
right_motor_handle = sim.getObject('/PioneerP3DX/rightMotor')
#
def steer(angle): #function angle adjustment
	if angle>5: #boleh calibrate
		sim.setJointTargetVelocity(left_motor_handle, -0.5)
		sim.setJointTargetVelocity(right_motor_handle, 0.5)	
	elif angle<-5:
		sim.setJointTargetVelocity(left_motor_handle, 0.5)
		sim.setJointTargetVelocity(right_motor_handle, -0.5)	
	else:
		return 1	
		
def distance(targetx,targety):
    # distance robot n target
    distance = math.sqrt((targetx - position_robot[0])**2 + (targety - position_robot[1])**2)
    return distance

def angle(targetx, targety):
    # Calculate vectors from robot to target and from robot to front
    vector_to_target = [targetx - position_robot[0], targety - position_robot[1]]
    vector_to_front = [position_front[0] - position_robot[0], position_front[1] - position_robot[1]]

    # Calculate the dot product between two vectors
    dot_product = vector_to_target[0] * vector_to_front[0] + vector_to_target[1] * vector_to_front[1]

    # Calculate magnitudes of vectors
    magnitude_target = math.sqrt(vector_to_target[0] ** 2 + vector_to_target[1] ** 2)
    magnitude_front = math.sqrt(vector_to_front[0] ** 2 + vector_to_front[1] ** 2)

    if magnitude_target == 0 or magnitude_front == 0:
        return 0  # Handle the case where one of the vectors has zero magnitude

    # Calculate angle using dot product and arccosine
    angle_radians = math.acos(dot_product / (magnitude_target * magnitude_front))

    # Convert angle from radians to degrees
    angle_degrees = math.degrees(angle_radians)

    # Calculate cross product to determine angle + or -
    cross_product = vector_to_target[0] * vector_to_front[1] - vector_to_target[1] * vector_to_front[0]

    # Compare calculated cross product +/- to determine +/- angle
    if cross_product < 0:
        return angle_degrees
    else:
        return -angle_degrees

def coord_to_int(coord): #function coord in coppelia turn to matrix position
    # verify coord in coppelia paramter
    #coord = max(-5, min(5, coord))
    
    # Map the coordinate to the nearest integer in range of 0 to 10
    mapped_value = int((coord + 4.6) )
    
    return mapped_value

def new_path(targetx,targety):
#Declare these variables as global to change the value globally, if not the variables will be considered only in the function
	global x_map, y_map, target_index 
	
	matrix = [
	[1, 1, 1, 1, 0, 1, 1, 0, 1, 1],   #last row and column not applicable, just for map calibration, set to 0\
	[1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
	[1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
	[0, 0, 1, 0, 0, 1, 1, 0, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
	[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
	[0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
	#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	]
		# create the grid
	grid = Grid(matrix=matrix)	
	target_index = 1
	#print(coord_to_int(position_robot[1]))
	#print(coord_to_int(position_robot[0]))
	start = grid.node(coord_to_int(position_robot[1]), coord_to_int(position_robot[0]))
	end = grid.node(coord_to_int(targetx), coord_to_int(targety))

	# Create a new AStarFinder instance each time
	finder = AStarFinder(diagonal_movement=DiagonalMovement.always)

	# Use finder to find path
	path, runs = finder.find_path(start, end, grid)
	# Check if the path is not empty before proceeding
	if path:
		# Extract x and y coordinates into separate arrays as integers
		y = [int(node.x) for node in path]
		x = [int(node.y) for node in path]

		print(grid.grid_str(path=path, start=start, end=end))

		# Define the mapping scaling factor for map coordinates
		scale_factor = 1  # 0.5 because the map spans from -2.5 to 2.5 over a 10x10 grid

		# Map the coordinates to your map's coordinate system
		x_map = [-4.6 + coord_x * scale_factor for coord_x in x]
		y_map = [-4.6 + coord_y * scale_factor for coord_y in y]

def move(condition):
	if condition == "straight":
		sim.setJointTargetVelocity(left_motor_handle, -1.5)
		sim.setJointTargetVelocity(right_motor_handle, -1.5)
	elif condition == "stop":
		sim.setJointTargetVelocity(left_motor_handle, 0)
		sim.setJointTargetVelocity(right_motor_handle, 0)
		
	
# arm function	
joint_names = [
    '/UR3/joint',  # joint alias
    '/UR3/link/joint',
    '/UR3/link/joint/link/joint',
    '/UR3/link/joint/link/joint/link/joint',
    '/UR3/link/joint/link/joint/link/joint/link/joint',
    '/UR3/link/joint/link/joint/link/joint/link/joint/link/joint',
]
joint_handles = []
for joint_name in joint_names:
    joint_handle = sim.getObject(joint_name)
    joint_handles.append(joint_handle)

def pos0():
	start_time = time.time()
	while time.time() - start_time < timer_duration:	
		sim.setJointTargetPosition(joint_handles[0], 0)
		sim.setJointTargetPosition(joint_handles[1], 0)
		sim.setJointTargetPosition(joint_handles[2], 0)
		sim.setJointTargetPosition(joint_handles[3], 0)
		sim.setJointTargetPosition(joint_handles[4], 0)
		sim.setJointTargetPosition(joint_handles[5], 0)
		sim.step()
def pos1():
	start_time = time.time()
	while time.time() - start_time < timer_duration:	
		sim.setJointTargetPosition(joint_handles[0], +12.66 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[1], -12.813 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[2], +116.343 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[3], -103.531 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[4], -12.66 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[5], 0 * (3.14159 / 180.0))
		sim.step()
def pos2():
	start_time = time.time()
	while time.time() - start_time < timer_duration:	
		sim.setJointTargetPosition(joint_handles[0], +38.333 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[1], -30.328 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[2], +126.956 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[3], -96.628 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[4], -38.333 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[5], 0 * (3.14159 / 180.0))
		sim.step()
def pos3():
	start_time = time.time()
	while time.time() - start_time < timer_duration:	
		sim.setJointTargetPosition(joint_handles[0], -12.66 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[1], +12.469 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[2], -117.44 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[3], +104.972 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[4], +12.66 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[5], 0 * (3.14159 / 180.0))
		sim.step()
def posPut():
	start_time = time.time()
	while time.time() - start_time < timer_duration:	
		sim.setJointTargetPosition(joint_handles[0], -51.677 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[1], -40 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[2], +126.956 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[3], -96.628 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[4], -38.333 * (3.14159 / 180.0))
		sim.setJointTargetPosition(joint_handles[5], 0 * (3.14159 / 180.0))
		sim.step()	

def put(obj):
	start_time = time.time()
	while time.time() - start_time < timer_duration:	
		sim.setObjectParent(obj,manipulator,0)
		sim.step()

def take(obj):
	start_time = time.time()
	while time.time() - start_time < timer_duration:	
		sim.setObjectParent(obj,tip,0)
		sim.step()

def image_process(lower_color, upper_color):
	global X,Y,cond
	# Get image data from vision sensor
	image_bytes, resolution = sim.getVisionSensorImg(sensor, 0, 0.0, [0, 0], [0, 0])
	# Convert image bytes to numpy array
	image_np = np.frombuffer(image_bytes, dtype=np.uint8)
	image_np = image_np.reshape((resolution[1], resolution[0], 3))
	# Flip the image vertically
	image_np = np.flipud(image_np)
	# Convert BGR to HSV
	hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
	# Convert BGR to RGB (not important, just to print the image in RGB)
	image_reference = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv_image, lower_color, upper_color)
	# Print the blue mask to check if any blue regions are detected
	#print("Blue Mask:", blue_mask)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Check if any blue contours are found
	if contours:
		# Draw contours on the image
		cv2.drawContours(image_reference, contours, -1, (0, 255, 0), 2)
		# Iterate through contours and find centroid of each
		for contour in contours:
			# Calculate centroid using moments
			M = cv2.moments(contour)
			if M["m00"] != 0:
				X = int(M["m10"] / M["m00"])
				Y = int(M["m01"] / M["m00"])
				if X<100:
					cond = 1
				elif X>100 and X<200:
					cond = 2
				elif X>200:
					cond = 3
				else:
					print("Chosen object not found!")
				
				#print(cond)
			else:
				X, Y = 0, 0
		  #  print("Coordinates:", X, Y)  # Print centroid coordinates
	# Display the image with contours
	cv2.imshow('Image with Contours', image_reference)
	key = cv2.waitKey(1)  # Wait 1 millisecond for a key press
	sim.step()	

# NLP function

def NER(command):
    global train
    if train:
        train = False
        global nlp2
        train = False
        nlp2 = spacy.blank("en")

        # Add entity labels for room and sphere color
        ner = nlp2.add_pipe("ner")
        ner.add_label("ROOM")
        ner.add_label("SPHERE_COLOR")

        # Prepare training data
        train = False
        training_data_room = [
            ("The meeting will take place in room 101.", {"entities": [(31, 39, "ROOM")]}),
            # Add more examples with corresponding entity annotations for rooms
        ]
        training_data_object = [
            ("Take red sphere.", {"entities": [(5, 8, "SPHERE_COLOR")]}),
            ("Take green sphere.", {"entities": [(5, 10, "SPHERE_COLOR")]}),
            ("Take blue sphere.", {"entities": [(5, 9, "SPHERE_COLOR")]}),
            # Add more examples with corresponding entity annotations for sphere color
        ]

        # Merge training data
        training_data = training_data_room + training_data_object

        # Train the NER model
        other_pipes = [pipe for pipe in nlp2.pipe_names if pipe != "ner"]
        with nlp2.disable_pipes(*other_pipes):
            nlp2.begin_training()
            for epoch in range(100):  # adjust the number of epochs as needed
                random.shuffle(training_data)
                for text, annotations in training_data:
                    doc = nlp2.make_doc(text)
                    nlp2.update([Example.from_dict(doc, annotations)], drop=0.5)

    # Test the custom NER model
    doc = nlp2(command)

    # Print recognized entities
    for ent in doc.ents:
        if ent.label_ == "ROOM" or ent.label_ == "SPHERE_COLOR":
            return ent.text


	# Print the entire doc object for additional debugging
	#print("Entire Doc Object:")
	#print(doc)
	

def process_command(command):
	#print(f"Recognized text: {command}")

	# Tokenize the command using spaCy
	doc = nlp(command)

	# Example: Extracting the verb and object
	verb = None
	obj = None
	num = None

	for token in doc:
		# Check for synonyms
		if (token.pos_ == "VERB" and token.dep_ == "ROOT") or (token.pos_ == "NOUN" and token.dep_ == "ROOT")or (token.pos_ == "ADJ" and token.dep_ == "ROOT"):
			verb = token.text
			#print(verb)
	for key,values in synonyms.items():
		if verb in values:
			verb = key

	#print(f"Extracted verb: {verb}")
	return(verb)

	

def perform_action(verb, ent):
	global act
	# Implement your logic for performing the action
	if verb == "go"  and ent:
		if ent == "room one":
			act = 1
		elif ent == "room two":
			act = 2
		elif ent == "room three":
			act = 3
		print(f"Performing action: {verb} to {ent}")
	elif verb == "take" and ent:
		if ent == "red":
			act = 4
		elif ent == "green":
			act = 5
		elif ent == "blue":
			act = 6
			print(f"{verb} {ent}")
	elif verb == "store":
		act = 7
	else:
		print("Sorry, I didn't understand")
		
act = None

global result, packet1, packet2,X,Y,cond,state
timer_duration = 1.6
cond = None
state = None
valid = None
X= None
Y= None
# Initialize index to first target coordinate
target_index = 1
x_map = []
y_map = []
text = None
while True:
	if keyboard.is_pressed('space'):
		# Reset recognized_text for each recording session
		recognized_text = ""

		start_time = time.time()
		print("Recording....")

		while time.time() - start_time <= record_duration:
			data = stream.read(8000)
			rec.AcceptWaveform(data)

		# Process the final result
		result = rec.Result()
		if result:
			result_dict = json.loads(result)
			recognized_text = result_dict.get("text", "")
			print("Recognized Text:", recognized_text)
			
			text=recognized_text
			verb = process_command(text)
			entity = NER(text)
			perform_action(verb, entity)
			if act == 1:
				new_path(position_alpha[1],position_alpha[0])
			elif act == 2:
				new_path(position_beta[1],position_beta[0])
			elif act == 3:
				new_path(position_gamma[1],position_gamma[0])

			elif act == 4:
				print("Pick Red")
				image_process(lower_red,upper_red)
				print("Coordinates:", X, Y)  # Print centroid coordinates
				valid = 1
			elif act == 5:
				print("Pick Green")
				image_process(lower_green,upper_green)
				print("Coordinates:", X, Y)  # Print centroid coordinates
				valid = 1
			elif act == 6:
				print("Pick Blue")
				image_process(lower_blue,upper_blue)
				print("Coordinates:", X, Y)  # Print centroid coordinates
				valid = 1
			elif act == 7:
				if valid == 1:
					cond = 4
			
				
			if cond == 1:
				pos1()
				take(obj1)
				pos0()
				posPut()
				put(obj1)
				pos0()
				cond = None
				state = obj1
			elif cond == 2:
				pos2()
				take(obj2)
				pos0()
				posPut()
				put(obj2)
				pos0()
				cond = None
				state = obj2
			elif cond == 3:
				pos3()
				take(obj3)
				pos0()
				posPut()
				put(obj3)
				pos0()
				cond = None
				state = obj3
			elif cond == 4:
				posPut()
				take(state)
				pos0()
				if state == obj1:
					pos1()
					put(obj1)
				elif state == obj2:
					pos2()
					put(obj2)
				elif state == obj3:
					pos3()
					put(obj3)	
				pos0()	
				cond = None
				state = None
				valid = None

			while target_index < len(x_map):
					position_front = sim.getObjectPosition(front)
					position_robot = sim.getObjectPosition(robot)
					position_alpha = sim.getObjectPosition(alpha)
					position_beta = sim.getObjectPosition(beta)
					position_gamma = sim.getObjectPosition(gamma)
					current_x = x_map[target_index]
					current_y = y_map[target_index]
					#print(target_index)
					dist = distance(current_x, current_y)

					if dist > 0.08:  # boleh calibrate
						if steer(angle(current_x, current_y)):
							move("straight")
					else:
						move("stop")
						# Move to next target coordinate
						target_index += 1		
					sim.step()



			sim.step()


