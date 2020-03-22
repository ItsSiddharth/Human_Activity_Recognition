from PIL import Image, ImageDraw
import cv2
import numpy as np
import posenet

MIN_CONFIDENCE = 0.40

if __name__ == '__main__':
	body_joints = [[posenet.BodyPart.LEFT_WRIST, posenet.BodyPart.LEFT_ELBOW],
	               [posenet.BodyPart.LEFT_ELBOW, posenet.BodyPart.LEFT_SHOULDER],
	               [posenet.BodyPart.LEFT_SHOULDER, posenet.BodyPart.RIGHT_SHOULDER],
	               [posenet.BodyPart.RIGHT_SHOULDER, posenet.BodyPart.RIGHT_ELBOW],
	               [posenet.BodyPart.RIGHT_ELBOW, posenet.BodyPart.RIGHT_WRIST],
	               [posenet.BodyPart.LEFT_SHOULDER, posenet.BodyPart.LEFT_HIP],
	               [posenet.BodyPart.LEFT_HIP, posenet.BodyPart.RIGHT_HIP],
	               [posenet.BodyPart.RIGHT_HIP, posenet.BodyPart.RIGHT_SHOULDER],
	               [posenet.BodyPart.LEFT_HIP, posenet.BodyPart.LEFT_KNEE],
	               [posenet.BodyPart.LEFT_KNEE, posenet.BodyPart.LEFT_ANKLE],
	               [posenet.BodyPart.RIGHT_HIP, posenet.BodyPart.RIGHT_KNEE],
	               [posenet.BodyPart.RIGHT_KNEE, posenet.BodyPart.RIGHT_ANKLE]]

	image = Image.open("./test1.jpg")
	draw = ImageDraw.Draw(image)

	# image = cv2.imread('test.jpg')
	# image = np.array(image)

	posenet = posenet.PoseNet(model_path="./posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite",
	                          image_path="./test1.jpg")
	person = posenet.estimate_pose()

	for line in body_joints:

		if person.keyPoints[line[0].value[0]].score > MIN_CONFIDENCE and person.keyPoints[line[1].value[0]].score > MIN_CONFIDENCE:
			start_point_x, start_point_y = int(person.keyPoints[line[0].value[0]].position.x), int(person.keyPoints[line[0].value[0]].position.y)
			end_point_x, end_point_y = int(person.keyPoints[line[1].value[0]].position.x), int(person.keyPoints[line[1].value[0]].position.y)
			draw.line((start_point_x, start_point_y, end_point_x, end_point_y),
			          fill=(255, 255, 0), width=3)

	for key_point in person.keyPoints:
		if key_point.score > MIN_CONFIDENCE:
			left_top_x, left_top_y = int(key_point.position.x) - 5, int(key_point.position.y) - 5
			right_bottom_x, right_bottom_y = int(key_point.position.x) + 5, int(key_point.position.y) + 5
			draw.ellipse((left_top_x, left_top_y, right_bottom_x, right_bottom_y),
			             fill=(0, 128, 0), outline=(255, 255, 0))

	print('total score : ', person.score)

	image = image.resize((600,600))
	image = np.array(image)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# image.save("./result.png")
	cv2.imshow('Result', image)
	cv2.waitKey(0)