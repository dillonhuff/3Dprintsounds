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

rad = 1.0
ts = [0, 0.1, 0.2, 0.3, 0.4]


for t in np.arange(0.0, 360, 1.0):
    xpt = rad*np.cos(t)
    ypt = rad*np.sin(t)

    print 'G1 F1800 X%f Y%f' % (xpt, ypt)
    plt.plot(rad*np.cos(t), rad*np.sin(t), 'bo')
    
#plt.show()
