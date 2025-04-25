import numpy as np

width = 1280  
height = 720

import time
start = time.time()

from graphicPipeline import GraphicPipeline

pipeline = GraphicPipeline(width,height)

from camera import Camera
from projection import Projection


position = np.array([7, 3, 4.8])
lookAt = np.array([-1.8, 0.4, -3.2])
up = np.array([0.33333333,  0.33333333, -0.66666667])
right = np.array([-0.57735027,  0.57735027,  0.])



################################################################################################# 1st chunck
lightPosition = np.array([10, 0, 10])
lightLookAt = np.array([0, 0, 0])
forward = lightLookAt - lightPosition
forward = forward / np.linalg.norm(forward)
lightUp = np.array([0, 1, 0])
lightRight = np.cross(lightUp, forward)
lightUp = np.cross(forward, lightRight)

lightCam = Camera(lightPosition, forward, lightUp, lightRight)
#################################################################################################

cam = Camera(position, lookAt, up, right)

nearPlane = 0.1
farPlane = 50.0
fov = 1.047
aspectRatio = width/height

proj = Projection(nearPlane ,farPlane,fov, aspectRatio) 

########################################################### 2nd chunck
lightNearPlane = 0.1
lightFarPlane = 20.0
lightFov = 1.047
lightProj = Projection(lightNearPlane, lightFarPlane, lightFov, aspectRatio)

lightViewMatrix = lightCam.getMatrix()
lightProjMatrix = lightProj.getMatrix()
###########################################################

import os
print("Current working directory:", os.getcwd())

from readply import readply

vertices, triangles = readply('Scene3.ply')

# load and show an image with Pillow
from PIL import Image
from numpy import asarray
# Open the image form working directory
image = asarray(Image.open('suzanne.png'))


data = dict([
  ('viewMatrix',cam.getMatrix()),
  ('projMatrix',proj.getMatrix()),
  ('cameraPosition',position),
  ('lightPosition',lightPosition),
  ('texture', image),
])

start = time.time()

################################################################### 3rd chunck
lightData = dict([
  ('viewMatrix',lightViewMatrix),
  ('projMatrix',lightProjMatrix),
  ('cameraPosition',lightPosition),
  ('lightPosition',lightPosition),
  
])

shadowPipeline = GraphicPipeline(1024, 1024)
shadowPipeline.draw(vertices, triangles, lightData, shade=False)
shadowMap = shadowPipeline.depthBuffer

data['lightViewMatrix'] = lightViewMatrix
data['lightProjMatrix'] = lightProjMatrix
data['shadowMap'] = shadowMap
#debug
import matplotlib.pyplot as plt
plt.imshow(shadowMap, cmap='gray')
plt.title("Shadow Map from Light View")
plt.show()

print("Shadow map min/max:", np.min(shadowMap), np.max(shadowMap))
###################################################################

pipeline.draw(vertices, triangles, data)

end = time.time()
print(end - start)

import matplotlib.pyplot as plt
imgplot = plt.imshow(pipeline.image)
plt.show()