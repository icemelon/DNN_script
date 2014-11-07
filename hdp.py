import os
import sys
import re
import argparse
import subprocess

from config import *
import scheduler

def create(jobName, socket):
	cmd = 'job new /scheduler:%s /jobtemplate:%s /numsockets:%s-%s /jobname:%s /runtime:%s' \
		% (HDP_SCHEDULER, HDP_JOBTEMP[0], socket, socket, jobName, HDP_JOBTEMP[1])
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	ret = p.stdout.read().split('\n')
	jobID = None
	for line in ret:
		if line.startswith('Created'):
			jobID = eval(line[line.index('ID:')+3:])
			break
	if (jobID == None):
		print 'Error in creating a new job'
		print '\n'.join(ret)
	return jobID

def addTask(jobID, socket, workdir, command, stdout):
	cmd = 'job add %s /scheduler:%s /workdir:%s /numsockets:%s-%s /stdout:%s %s' \
		% (jobID, HDP_SCHEDULER, workdir, socket, socket, stdout, command)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	ret = p.stdout.read()
	if 'added' in ret:
		return True
	else:
		print ret
		return False

def submit(jobID):
	cmd = 'job submit /scheduler:%s /id:%s' % (HDP_SCHEDULER, jobID)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	ret = p.stdout.read()
	if 'submitted' in ret:
		return True
	else:
		print ret
		return False

def execute(jobName, socket, workdir, command, stdout):
	jobID = create(jobName, socket)
	if jobID is None:
		return False
	print "Created job %s" % jobID

	if not addTask(jobID, socket, workdir, command, stdout):
		return False
	print "Added task \"%s\" (stdout: %s)" % (command, stdout)
	
	if not submit(jobID):
		return False
	print "Submitted job %s" % jobID

def finish(jobName):
	cmd = 'job list /scheduler:%s /jobname:%s' % (HDP_SCHEDULER, jobName)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	ret = p.stdout.read()
	if 'Running' in ret or 'Queued' in ret:
		return False
	else:
		return True

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Using TLC to train neural networks')
	parser.add_argument('dataset', help='Working directory that contains dataset')
	parser.add_argument('logfile', help='Log file (header should be initialed)')
	parser.add_argument('-s', '--socket', type=int, default=1, help='Available socket number (Default: 1)')
	parser.add_argument('--tlc', default=HDP_TLC_PATH, help='TLC executable path')

	try:
		args = parser.parse_args()
	except:
		exit()

	if not os.path.exists(args.dataset):
		print 'Working directory "%s" doesn\'t exist' % args.dataset
		exit()
	if not os.path.exists(args.logfile):
		print 'Log file "%s" doesn\'t exist' % args.logfile
		exit()
	print 'JobTemplate = %s' % str(HDP_JOBTEMP)
	print 'Socket = %s' % args.socket

	# let's replace Z:\ by HPD_ROOT_DIR
	workdir = os.path.join(HDP_ROOT_DIR, os.getcwd()[3:])

	script = os.path.relpath(scheduler.__file__)
	command = "python %s %s %s -s %s --tlc %s" % \
		(script, args.dataset, args.logfile, args.socket, args.tlc)
	stdout = args.logfile[:-4] + ".out"

	dataset = os.path.basename(args.dataset)
	threadName = os.path.basename(args.logfile)
	threadName = threadName[:threadName.rfind('.')]
	jobName = "%s_%s" % (dataset, threadName)

	execute(jobName, args.socket, workdir, command, stdout)
