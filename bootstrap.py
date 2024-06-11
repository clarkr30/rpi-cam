# Imports
import mediapipe as mp
from picamera2 import Picamera2
import time
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})
pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)

def draw_pose(image, landmarks):
	''' 
	TODO Task 1
	
	Code to this fucntion to draw circles on the landmarks and lines
	connecting the landmarks then return the image.
	
	Use the cv2.line and cv2.circle functions.

	landmarks is a collection of 33 dictionaries with the following keys
		x: float values in the interval of [0.0,1.0]
		y: float values in the interval of [0.0,1.0]
		z: float values in the interval of [0.0,1.0]
		visibility: float values in the interval of [0.0,1.0]
		
	References:
	https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
	https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
	'''

	# copy the image
	landmark_image = image.copy()
	
	# get the dimensions of the image
	height, width, _ = image.shape

	mp_drawing.draw_landmarks(
        landmark_image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

	return landmark_image

def main():
	''' 
	TODO Task 2
		modify this fucntion to take a photo uses the pi camera instead 
		of loading an image
	'''
	
	image = pi_camera.capture_array()
	cv2.imwrite('input.png', image)


	'''
	TODO Task 3
		modify this function further to loop and show a video
	'''
	cap = cv2.VideoCapture('test.mp4')
	imgs = []
	vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameNum = 0

	height = -1
	width = -1

	# Create a pose estimation model 
	mp_pose = mp.solutions.pose
	
	# start detecting the poses
	with mp_pose.Pose(
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as pose:

			# load test image
			# image = cv2.imread("input.png")

			# To improve performance, optionally mark the image as not 
			# writeable to pass by reference.
			image.flags.writeable = False
			
			while cap.isOpened():
				ret, frame = cap.read()
				if not ret:
					break

				print("Processing Frame #" + str(frameNum) + "/" + str(vid_length))

				image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

				if(height < 0 or width < 0):
					height,width, = image[1].shape

				# get the landmarks
				results = pose.process(image)
				
				if results.pose_landmarks != None:
					result_image = draw_pose(image, results.pose_landmarks)
					cv2.imwrite('output.png', result_image)
					imgs.append(result_image)
					# print(results.pose_landmarks)
				else:
					# print('No Pose Detected')
					imgs.append(image)
				frameNum += 1
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			video=cv2.VideoWriter('outputVideo.mp4',fourcc,15,(width,height))
			for i in range(len(imgs)):
				video.write(imgs[i])
			video.release()

if __name__ == "__main__":
	main()
	print('done')
