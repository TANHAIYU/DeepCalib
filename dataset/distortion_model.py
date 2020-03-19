import math
import numpy as np
import random


"""" generate distortion parameter for different distortion model """
def distortionParameter():
    parameters = []
    # Lambda = np.random.random_sample( )*1 #-5e-5/4
    fx=random.uniform(718.856*0.7, 718.8562*1.3)  #0.6-1.8
    fy=random.uniform(718.856*0.7, 718.8562*1.3)  #0.6-1.8
    x0 =random.uniform(607.1928*0.45, 607.1928*0.55) #try
    y0 = random.uniform(92.60785*0.45, 92.60785*0.55)
    Lambda = (2 * random.uniform(0,1) -1) * 0.8 # vor- keine 0.8
    parameters.append(Lambda)
    parameters.append(x0)
    parameters.append(y0)
    parameters.append(fx)
    parameters.append(fy)
    return parameters

def distortionModel(xd, yd, W, H, parameter):
    Lambda = parameter[0] # distortion parameter
    x0    = parameter[1]  # princip point - Cx
    y0    = parameter[2]  # princip point - Cy
    f    = parameter[3]  # horizont focal length
    X=(xd-x0)/f
    Y=(yd-y0)/f
    r2 = X*X + Y*Y

    coeff = 1 + Lambda * r2
    newX = X*coeff
    newY = Y*coeff

    # calculate distortion coefficent
    if (coeff == 0):
        xu = W
        yu = H
    else:
    # add distortion effect and transform to image coordinate
        xu = f*newX + x0
        yu = f*newY + y0
        #xu = newX + x0
        #yu = newY + y0
    return xu, yu
