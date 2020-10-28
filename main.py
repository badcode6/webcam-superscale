from __future__ import print_function
import datetime
import tensorflow as tf
import cv2
import pyvirtualcam
from PIL import Image
import numpy as np
from threading import Thread
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config=config)

model = tf.keras.models.load_model("esrgan")

class WebcamVideoStream:
	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False
	def start(self):
		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		while True:
			if self.stopped:
				return
			(self.grabbed, self.frame) = self.stream.read()
	def read(self):
		return self.frame
	def stop(self):
		self.stopped = True

cammode = True

if cammode:
    cam = pyvirtualcam.Camera(width=1600, height=1200, fps=30)

vs = WebcamVideoStream(src=0).start()
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	deff = np.array([np.array(frame).astype(np.float32)])
	super_resolution = model.predict(deff)
	super_resolution = tf.cast(tf.clip_by_value(super_resolution[0], 0, 255), tf.uint8)
	frmm = np.array(super_resolution)
	if cammode:
		b_channel, g_channel, r_channel = cv2.split(frmm)
		alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50
		img_RGBA = cv2.merge((r_channel, g_channel, b_channel, alpha_channel))
		cam.send(img_RGBA)
		cam.sleep_until_next_frame()
	else:
		cv2.imshow('frame',frmm)
		cv2.waitKey(1)


cv2.destroyAllWindows()
vs.stop()