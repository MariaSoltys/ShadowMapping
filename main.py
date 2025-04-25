import numpy as np

width = 640
height = 360

import time
start = time.time()


from graphicPipeline import GraphicPipeline


pipeline = GraphicPipeline(width,height)


from camera import Camera
from projection import Projection


position = np.array([3.4, 2.0, 4.8])
lookAt = np.array([-0.577, -0.577, -0.577])
up = np.array([0.33333333,  0.33333333, -0.66666667])
right = np.array([-0.57735027,  0.57735027,  0.])

lightPosition = np.array([10, 0, 10])
lightLookAt = np.array([0, 0, 0])
forward = lightLookAt - lightPosition
forward = forward / np.linalg.norm(forward)
lightUp = np.array([0, 1, 0])
lightRight = np.cross(lightUp, forward)
lightUp = np.cross(forward, lightRight)

lightCam = Camera(lightPosition, forward, lightUp, lightRight)


cam = Camera(position, lookAt, up, right)

nearPlane = 0.1
farPlane = 50.0
fov = 1.047
aspectRatio = width/height

proj = Projection(nearPlane ,farPlane,fov, aspectRatio) 


lightNearPlane = 0.1
lightFarPlane = 20.0
lightFov = 1.047
lightProj = Projection(lightNearPlane, lightFarPlane, lightFov, aspectRatio)


lightViewMatrix = lightCam.getMatrix()
lightProjMatrix = lightProj.getMatrix()

import os
print("Current working directory:", os.getcwd())

from readply import readply

vertices, triangles = readply('tree.ply')

from PIL import Image
from numpy import asarray
image = asarray(Image.open('terrain.jpg'))


data = dict([
  ('viewMatrix',cam.getMatrix()),
  ('projMatrix',proj.getMatrix()),
  ('cameraPosition',position),
  ('lightPosition',lightPosition),
  ('texture', image),
])

start = time.time()

lightData = dict([
    ('viewMatrix', lightViewMatrix),
    ('projMatrix', lightProjMatrix),
    ('cameraPosition', lightPosition),
    ('lightPosition', lightPosition),
    ('lightNearPlane', lightNearPlane),
    ('lightFarPlane', lightFarPlane),
])

shadowPipeline = GraphicPipeline(width, height)
shadowPipeline.draw(vertices, triangles, lightData, shade=False)
shadowMap = shadowPipeline.depthBuffer

data['lightViewMatrix'] = lightViewMatrix
data['lightProjMatrix'] = lightProjMatrix
data['shadowMap'] = shadowMap

data['lightNearPlane'] = lightNearPlane
data['lightFarPlane'] = lightFarPlane

import matplotlib.pyplot as plt
plt.imshow(shadowMap, cmap='gray')
plt.title("Shadow Map from Light View")
plt.show()

print("Shadow map min/max:", np.min(shadowMap), np.max(shadowMap))

pipeline.draw(vertices, triangles, data)

end = time.time()
print(end - start)

import matplotlib.pyplot as plt
imgplot = plt.imshow(pipeline.image)
plt.show()