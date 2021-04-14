import numpy as np
import cv2
from math import floor
from glob import glob
from PIL import Image


from sph import SPHSystem
from body import Body


_WIDTH  = 200
_HEIGHT = 200
_FRAMECOUNT = 300
_FRAMEDURATION = 16 #~60 fps assuming little processing time


def render_sph():
    '''
    Render and run an SPH system.
    '''

    # initialise SPH
    N = 1500
    masses = np.full((1, N), 1/N)
    positions = np.vstack((np.random.rand(N), np.random.rand(N) * 0.25))
    bodies = [Body(1.0, np.array([0.5, 0.5])), Body(10.0, np.array([0.25, 0.5]))]#, Body(0.00001, np.array([0.75, 0.5]))]
    system = SPHSystem(masses, positions, bodies)

    # initialize transform matrix
    # transforms particles positions from x,y ∈ [0,1] to x,y ∈ [_WIDTH, _HEIGHT]
    transform = np.array([[_WIDTH, 0], [0, _HEIGHT]])

    frame = 0

    # main rendering loop
    while(frame < _FRAMECOUNT):
        # initialise image
        img = np.full((_WIDTH, _HEIGHT, 3), 255, np.uint8)

        # get the particle coordinates
        coordinates = np.rint(transform @ system.position).astype(int)

        # draw particles in image
        for c in range(coordinates.shape[1]):
            x, y = coordinates[0, c], coordinates[1, c]
            if (x < 0 or y < 0 or x >= _WIDTH or y >= _HEIGHT):
                print(f"Particle broke canvas bounderies! x={x} y={y}")
                continue

            img[_HEIGHT - y - 1, x] = [255, 0, 0]

        # apply post processing to water
        img = _post_process(img)

        # draw the objects
        for b in system.body:
            cv2.circle(img, (floor(b.c[0] * _WIDTH), _HEIGHT - floor(b.c[1] * _HEIGHT)), floor(_WIDTH/10), (0,0,255), cv2.FILLED)

        #write image
        cv2.imwrite(f'imgcache/{frame}.png', img)
        frame += 1
              
        # show image
        print(f"time={system.time}")
        cv2.imshow('SPH System', img)
        cv2.waitKey(_FRAMEDURATION) 
    
        # update the SPH system
        system.update()

    frames=[Image.open(f'imgcache/{frame}.png') for frame in range(_FRAMECOUNT)]   
    frames[0].save('png_to_gif.gif', format='GIF', append_images=frames[1:], save_all=True, duration=_FRAMEDURATION, loop=0)     
   

def _post_process(img):
    '''
    TODO doc
    '''

    kernel = np.array([[0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0]], np.uint8)
    img = cv2.erode (img, kernel, iterations=4)
    img = cv2.dilate(img, kernel, iterations=3)

    return img


if __name__ == '__main__':
    render_sph()
   
