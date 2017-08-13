import numpy as np
import matplotlib.pyplot as plt

T = [1, 10, 20, 30, 40, 50, 60]
R = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]



def rtpairs(r, n):

    for i in range(len(r)):
       for j in range(n[i]):    
        yield r[i], j*(2 * np.pi / n[i])

pairs = rtpairs(R, T)

#print 'Length = ', len(pairs)

# for r, t in rtpairs(R, T):
#     print r, ' ', t
#     plt.plot(r * np.cos(t), r * np.sin(t), 'bo')

rad = 40
ts = [0, 0.1, 0.2, 0.3, 0.4]

center_x = 60
center_y = 60

increment = (2*np.pi) / 360


def move_increments(increment, wait_time_milliseconds):
    i = 1
    for t in np.arange(0.0, 2*np.pi, increment): #(2*np.pi) / 360):
        xpt = rad*np.cos(t)
        ypt = rad*np.sin(t)

        deg = (180.0 / np.pi) * t

        print 'G0 F3600 Z0.4'
        print 'G0 F3600 X%f Y%f' % (center_x, center_y)
        print 'G0 F3600 Z0.3'
        print 'G4 P%f' % (wait_time_milliseconds)
        print 'G1 F1800 X%f Y%f E0.01314 ; Line %d, with angle %f' % (center_x + xpt, center_y + ypt, i, deg)
        print 'G4 P%f' % (wait_time_milliseconds)

        i += 1

def deg_to_rad(deg):
    return (deg*np.pi) / 180

increment = deg_to_rad(45)
move_increments(increment, 2000)
