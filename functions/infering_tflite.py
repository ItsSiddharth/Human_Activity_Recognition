from PIL import Image, ImageDraw
import cv2
import numpy as np


from functions.posenet import BodyPart, PoseNet


MIN_CONFIDENCE = 0.40

def Key_Point_Generator(image):
	map_cord_to_part=[]
	body_joints = [[BodyPart.LEFT_WRIST,BodyPart.LEFT_ELBOW],
	               [BodyPart.LEFT_ELBOW,BodyPart.LEFT_SHOULDER],
	               [BodyPart.LEFT_SHOULDER,BodyPart.RIGHT_SHOULDER],
	               [BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW],
	               [BodyPart.RIGHT_ELBOW,BodyPart.RIGHT_WRIST],
	               [BodyPart.LEFT_SHOULDER,BodyPart.LEFT_HIP],
	               [BodyPart.LEFT_HIP,BodyPart.RIGHT_HIP],
	               [BodyPart.RIGHT_HIP,BodyPart.RIGHT_SHOULDER],
	               [BodyPart.LEFT_HIP,BodyPart.LEFT_KNEE],
	               [BodyPart.LEFT_KNEE,BodyPart.LEFT_ANKLE],
	               [BodyPart.RIGHT_HIP,BodyPart.RIGHT_KNEE],
	               [BodyPart.RIGHT_KNEE,BodyPart.RIGHT_ANKLE]]

	image = Image.fromarray(image, 'RGB')
	draw = ImageDraw.Draw(image)

		# image = cv2.imread('test.jpg')
		# image = np.array(image)

	Posenet = PoseNet(model_path="functions/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite",
		                          image=image)
	person, heatmaps = Posenet.estimate_pose()
		# print((len(person.keyPoints)))

	for line in body_joints:

		if person.keyPoints[line[0].value[0]].score > MIN_CONFIDENCE and person.keyPoints[line[1].value[0]].score > MIN_CONFIDENCE:
			start_point_x, start_point_y = int(person.keyPoints[line[0].value[0]].position.x), int(person.keyPoints[line[0].value[0]].position.y)
				# print(person.keyPoints[line[0].value[0]].position.x, person.keyPoints[line[0].value[0]].position.y)
			end_point_x, end_point_y = int(person.keyPoints[line[1].value[0]].position.x), int(person.keyPoints[line[1].value[0]].position.y)
			draw.line((start_point_x, start_point_y, end_point_x, end_point_y),
			          fill=(255, 255, 0), width=3)

	for key_point in person.keyPoints:
		if key_point.score > MIN_CONFIDENCE:
			left_top_x, left_top_y = int(key_point.position.x) - 5, int(key_point.position.y) - 5
			right_bottom_x, right_bottom_y = int(key_point.position.x) + 5, int(key_point.position.y) + 5
			draw.ellipse((left_top_x, left_top_y, right_bottom_x, right_bottom_y),
				             fill=(0, 128, 0), outline=(255, 255, 0))
			centre_x = (left_top_x+right_bottom_x)//2
			centre_y = (left_top_y+right_bottom_y)//2
			# print(centre_x, centre_y, str(key_point.bodyPart)[9:])
			# key_point_holder['{}'.format(key_point.BodyPart)]
			map_cord_to_part.append([centre_x, centre_y, str(key_point.bodyPart)[9:]])

		# print('total score : ', person.score)
	image = image.resize((600,600))
	image = np.array(image)
	return image, map_cord_to_part, heatmaps