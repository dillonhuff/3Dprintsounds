;FLAVOR:RepRap
;TIME:711
;Filament used: 0.153629m
;Layer height: 0.1
;Generated with Cura_SteamEngine 2.5.0
M190 S60
M104 S200
M109 S200
G28 ;Home
G1 Z15.0 F6000 ;Move the platform down 15mm
;Prime the extruder
G92 E0
G1 F200 E3
G92 E0
;LAYER_COUNT:58
;LAYER:0
M107
G0 F3600 X50.0 Y50.0 Z0.3
;TYPE:SKIRT
G1 F1800 X90.0 E0.01314
G1 F1800 Y90 E0.01314
G1 F1800 X50.0 E0.01314
G1 F1800 Y50 E0.01314

;Finish sequence
M107
M104 S0
M140 S0
;Retract the filament
G92 E1
G1 E-1 F300
G28 X0 Y0
M84
M104 S0
;End of Gcode
;SETTING_3 {"global_quality": "[general]\\nversion = 2\\nname = empty\\ndefiniti
;SETTING_3 on = custom\\n\\n[metadata]\\nquality_type = normal\\ntype = quality_
;SETTING_3 changes\\n\\n[values]\\n\\n"}
