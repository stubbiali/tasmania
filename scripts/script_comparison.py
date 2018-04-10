import numpy as np
import os
import pickle

#
# Mandatory settings
#
filename1 = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_kessler_wrf_clipping_maccormack.pickle')
filename2 = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_kessler_wrf_clipping_sedimentation_maccormack.pickle')
field = 'mass_fraction_of_precipitation_water_in_air'

#
# Load
#
with open(filename1, 'rb') as data:
	state_save1 = pickle.load(data)
	#field1 = state_save[field]
with open(filename2, 'rb') as data:
	state_save2 = pickle.load(data)
	#field2 = state_save[field]

print('here we are')

