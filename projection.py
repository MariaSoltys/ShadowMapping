import numpy as np

class Projection:
  def __init__(self, near,far,fov,aspectRatio) :
    self.nearPlane = near
    self.farPlane = far
    self.fov = fov
    self.aspectRatio = aspectRatio


  def getMatrix(self) :
    f = self.farPlane
    n = self.nearPlane
    s = 1/np.tan(self.fov/2)
    perspective = np.array([
      [s/self.aspectRatio,0,0,0],
      [0,s,0,0],
      [0,0,f/(f-n),-(f*n)/(f-n)],
      [0,0,1,0]
    ])

    return perspective

#############################################################3
class OrthographicProjection:
    def __init__(self, left, right, bottom, top, near, far):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far

    def getMatrix(self):
        l = self.left
        r = self.right
        b = self.bottom
        t = self.top
        n = self.near
        f = self.far

        return np.array([
            [2/(r - l), 0, 0, -(r + l)/(r - l)],
            [0, 2/(t - b), 0, -(t + b)/(t - b)],
            [0, 0, 2/(n - f), (f + n)/(n - f)],
            [0, 0, 0, 1]
        ])
