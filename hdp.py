import os
import sys
import re
import argparse
import subprocess

from config import *
import scheduler

def create(jobName, socket):
	cmd = 'job new /scheduler:%s /jobtemplate:%s /numsockets:%s-%s /jobname:%s /runtime:%s' \
		% (SCHEDULER, JOBTEMP[0], socket, socket, jobName, JOBTEMP[1])
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
		% (jobID, SCHEDULER, workdir, socket, socket, stdout, command)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	ret = p.stdout.read()
	if 'added' in ret:
		return True
	else:
		print ret
		return False

def submit(jobID):
	cmd = 'job submit /scheduler:%s /id:%s' % (SCHEDULER, jobID)
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
	cmd = 'job list /scheduler:%s /jobname:%s' % (SCHEDULER, jobName)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	ret = p.stdout.read()
	if 'Running' in ret or 'Queued' in ret:
		return False
	else:
		return True

"""
class HDPJob(object):
	def __init__(self, maxSocket):
		self.maxSocket = maxSocket
		self.jobId = None

	def create(self, taskName, socket):
		self.jobName = taskName
		cmd = 'job new /scheduler:%s /jobtemplate:%s /numsockets:%s-%s /jobname:%s' % (SCHEDULER, JOBTEMP, socket, socket, taskName)
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
		ret = p.stdout.read().split('\n')
		for line in ret:
			if line.startswith('Created'):
				self.jobId = eval(line[line.index('ID:')+3:])
				break
		if (self.jobId == None):
			print 'Error in creating a new job'
			print '\n'.join(ret)
			return False
		else:
			return True

	def addTask(self, task):
		cmd = 'job add %s /scheduler:%s /workdir:%s /stdout:%s %s @%s' % (self.jobId, SCHEDULER, os.path.join(ROOT_DIR, task.dataset), task.stdout, TLC_PATH, task.rspFile)
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
		ret = p.stdout.read()
		if 'added' in ret:
			return True
		else:
			return False

	def submit(self):
		cmd = 'job submit /scheduler:%s /id:%s' % (SCHEDULER, self.jobId)
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
		ret = p.stdout.read()
		if 'submitted' in ret:
			return True
		else:
			return False

	def execute(self, jobName):
		if isinstance(task, Task):
			if not self.create(task.taskName, 1):
				return False
			self.addTask(task)
		elif isinstance(task, ForkTask):
			if not self.create(task.taskName, min(self.maxSocket, task.taskCount)):
				return False
			for t in task.taskList:
				self.addTask(t)
		else:
			return False

		cnt = 0
		succ = False
		while cnt < 3:
			if self.submit():
				succ = True
				break
			cnt += 1
		return succ

	def finish(self):
		cmd = 'job list /scheduler:%s /jobname:%s' % (SCHEDULER, self.jobName)
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
		ret = p.stdout.read()
		if 'Running' in ret or 'Queued' in ret:
			return False
		else:
			return True
"""

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
	print 'JobTemplate = %s' % str(JOBTEMP)
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
