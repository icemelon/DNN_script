import os
import argparse
import subprocess

import scheduler

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Using TLC to train neural networks')
	parser.add_argument('taskfile', type=file, help='A file that contains all the tasks, each for one line')

	try:
		args = parser.parse_args()
	except:
		exit()
	
	tasks = args.taskfile.readlines()
	args.taskfile.close()
	script = os.path.relpath(scheduler.__file__)

	for task in tasks:
		task = task.strip()
		tokens = task.split()
		for t in tokens:
			if t.endswith(".log"):
				logfile = t
		stdout = os.path.splitext(logfile)[0] + ".out"
		# stderr = os.path.splitext(logfile)[0] + ".err"
		command = "python %s %s > %s" % (script, task, stdout)
		print "Start to run: %s" % command
		proc = subprocess.Popen(command, shell=True)
		proc.wait()