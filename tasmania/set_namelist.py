import os
import shutil

def set_namelist(user_namelist = None):
	"""
	Place the user-defined namelist module in the Python search path.
	This is achieved by physically copying the content of the user-provided module into TASMANIA_ROOT/namelist.py.

	Parameters
	----------
	user_namelist : str 
		Path to the user-defined namelist. If not specified, the default namelist TASMANIA_ROOT/_namelist.py is used.
	"""
	try:
		tasmania_root = os.environ['TASMANIA_ROOT']
	except RuntimeError:
		print('Hint: has the environmental variable TASMANIA_ROOT been set?')
		raise

	if user_namelist is None: # Default case
		src_file = os.path.join(tasmania_root, '_namelist.py')
		dst_file = os.path.join(tasmania_root, 'namelist.py')
		shutil.copy(src_file, dst_file)
	else:
		src_dir = os.curdir
		src_file = os.path.join(src_dir, user_namelist)
		dst_file = os.path.join(tasmania_root, 'namelist.py')
		shutil.copy(src_file, dst_file)
										
