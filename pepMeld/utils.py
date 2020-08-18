from datetime import datetime
import logging
import tempfile
import os

# import logger
log = logging.getLogger(__name__)

def print_status(status):
	'''print timestamped status update'''
	print('--[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] ' + status + '--')
	log.info(status)
	
def create_temp_folder():
	'''make temporary folder at specified location'''
	
	TMP_DIR = tempfile.mkdtemp()
	return TMP_DIR

def close_temp_folder(tmp_dir):
	'''destroy temporary folder after it is no longer used'''
	os.removedirs(tmp_dir)
	
def create_output_folder(cwd):
	'''create timestamped output folder at specified location'''
	
	# fetch current time
	CURRENT_TIME = datetime.now().strftime("%Y%m%d%H%M%S")

	# path to output folder
	OUTPUT_FOLDER = cwd + '/' + CURRENT_TIME

	# create folder if it doesn't already exist
	if not os.path.exists(OUTPUT_FOLDER):
		os.makedirs(OUTPUT_FOLDER)

	# print output folder name
	print_status('Output folder: ' + OUTPUT_FOLDER)
	
	return OUTPUT_FOLDER

def run_command(cmd_list, stdout_file = None, stderr_file = None, shell =True):
	'''run command with subprocess.call
	if stdout or stderr arguments are passed, save to specified file
	'''
	
	import subprocess
	
	print_status(' '.join(cmd_list)) # print status
	
	# if neither stdout or stderr specified
	if stdout_file is None and stderr_file is None:
		print(cmd_list)
		subprocess.call(cmd_list, shell = shell)
	 
	# if only stdout is specified
	elif stdout_file is not None and stderr_file is None:
		with open(stdout_file, 'w') as so:
			subprocess.call(cmd_list, stdout = so, shell = shell)
	 
	# if only stderr is specified
	elif stdout_file is None and stderr_file is not None:
		with open(stderr_file, 'w') as se:
			subprocess.call(cmd_list, stderr = se, shell = shell)
	 
	# if both stdout and stderr are specified
	elif stdout_file is not None and stderr_file is not None:
		with open(stdout_file, 'w') as so:
			with open(stderr_file, 'w') as se:
				subprocess.call(cmd_list, stdout = so, stderr = se, shell = shell)
	
	else: pass

def test_executable(cmd):
	'''check that a particular command can be run as an executable'''
	
	import shutil
	
	assert shutil.which(cmd) is not None, 'Executable ' + cmd + ' cannot be run'

def get_notebook_path(out_dir, main_dir, notebook_name,experiment):
	'''get name of  20835-genotyping.ipynb file in current working directory
	copy to output folder
	'''
	
	import os
	import shutil
	
	cwd = os.getcwd() # get working directory
	#notebook_path = cwd + '/20835-miseq-genotyping.ipynb' 
	notebook_path = main_dir + '/' + notebook_name + '.ipynb'
	# copy to output folder
	shutil.copy2(notebook_path, out_dir + '/' + str(experiment) + '.ipynb')
	print('Copied Jupyter Notebook to : ' + out_dir + '/' + str(experiment) + '.ipynb')

def copy_file_to_results(out_dir, filepath_list):
	'''get name of  20835-genotyping.ipynb file in current working directory
	copy to output folder
	'''
	
	import os
	import shutil
	
	for filepath_i in filepath_list:
		basefilename = os.path.basename(filepath_i)
		shutil.copy2(filepath_i, out_dir + '/'+ basefilename)
		print('Copied '+ basefilename + ' file to : ' + filepath_i)
	
def file_size(f):
	'''return file size'''
	import os
	
	return os.stat(f).st_size

def create_folder(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
		print('Created_Folder: ' + directory)
	else:
		print('Results Foler already exists: ' + directory)
	return()
def test_dir(directory,shortname):
	if not os.path.exists(directory):
		print( directory + 'directory does not exist for variable: ' +shortname+' ;. Script has terminated')
		raise RuntimeError(directory + 'directory does not exist for variable: ' +shortname+' ;. Script has terminated')
	return
	