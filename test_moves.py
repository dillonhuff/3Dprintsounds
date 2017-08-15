import numpy as np
import matplotlib.pyplot as plt

prelude = ';FLAVOR:RepRap\n;TIME:711\n;Filament used: 0.153629m\n;Layer height: 0.1\n;Generated with Cura_SteamEngine 2.5.0\nM190 S60\nM104 S200\nM109 S200\nG28 ;Home\nG1 Z15.0 F6000 ;Move the platform down 15mm\n;Prime the extruder\nG92 E0\nG1 F200 E3\nG92 E0\n;LAYER_COUNT:58\n;LAYER:0\nM107\n'

postscript = ';Finish sequence\nM107\nM104 S0\nM140 S0\n;Retract the filament\nG92 E1\nG1 E-1 F300\nG28 X0 Y0\nM84\nM104 S0\n;End of Gcode\n'

rad = 40


center_x = 60
center_y = 60

def deg_to_rad(deg):
    return (deg*np.pi) / 180

def rad_to_deg(deg):
    return (180.0 / np.pi) * deg

increment = deg_to_rad(10) #(2*np.pi) / 360

print 'Increment = ', increment

def move_increments(increment, wait_time_milliseconds):
    i = 1
    for t in np.arange(0.0, 2*np.pi, increment): #(2*np.pi) / 360):
        xpt = rad*np.cos(t)
        ypt = rad*np.sin(t)

        deg = rad_to_deg(t) #(180.0 / np.pi) * t

        print 'G0 F3600 Z0.4'
        print 'G0 F3600 X%f Y%f' % (center_x, center_y)
        print 'G0 F3600 Z0.3'
        print 'G4 P%d' % (wait_time_milliseconds)
        print 'G1 F1800 X%f Y%f E0.01314 ; Line %d, with angle %f' % (center_x + xpt, center_y + ypt, i, deg)
        print 'G4 P%d' % (wait_time_milliseconds)

        i += 1

increment = deg_to_rad(45)

print prelude
move_increments(increment, 3000)
print
print postscript
