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

for line in body_joints:
	person.keyPoints[line[0].value[0]]