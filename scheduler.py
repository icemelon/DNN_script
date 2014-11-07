import os
import sys
import time
import subprocess
import argparse

from task import *
from config import *
from logger import Logger
from trainer import *

class Scheduler(object):
	def __init__(self, maxSocket, tlcpath):
		self.maxSocket = maxSocket
		self.tlcpath = tlcpath
		self.pending = []
		self.running = {}

	def log(self, msg):
		sys.stdout.write("[%s] %s\n" % (time.asctime(), msg))
		sys.stdout.flush()

	def updateAndRun(self):
		finished = []
		for task in self.running:
			proc = self.running[task]
			if proc.poll() is not None:
				self.log("Task %s finished" % task.taskName)
				finished.append(task)

		for task in finished:
			del self.running[task]

		while len(self.pending) > 0 and len(self.running) < self.maxSocket:
			task = self.pending.pop(0)
			cmd = "%s @%s > %s" % (self.tlcpath, task.rspFile, task.stdout)
			proc = subprocess.Popen(cmd, cwd=task.rootdir, shell=True, universal_newlines=True)
			self.running[task] = proc
			self.log("Task %s is running" % task.taskName)

	def execute(self, task):
		if isinstance(task, Task):
			self.pending.append(task)
		elif isinstance(task, ForkTask):
			self.pending.extend(task.taskList)
		else:
			return False
		self.updateAndRun()
		return True

	def checkFinish(self, task):
		self.updateAndRun()
		if isinstance(task, Task):
			if task in self.pending or task in self.running:
				return False
			else:
				return True
		elif isinstance(task, ForkTask):
			for t in task.taskList:
				if t in self.pending or t in self.running:
					return False
			return True


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Using TLC to train neural networks')
	parser.add_argument('dataset', help='Working directory that contains dataset')
	parser.add_argument('logfile', help='Log file (header should be initialed)')
	parser.add_argument('-s', '--socket', type=int, default=1, help='Available socket number (Default: 1)')
	parser.add_argument('--tlc', default=LOCAL_TLC_PATH, help='TLC executable path')

	try:
		args = parser.parse_args()
	except:
		exit()

	args.dataset = os.path.abspath(args.dataset)
	args.logfile = os.path.abspath(args.logfile)

	if not os.path.exists(args.dataset):
		print 'Working directory "%s" doesn\'t exist' % args.dataset
		exit()
	if not os.path.exists(args.logfile):
		print 'Log file "%s" doesn\'t exist' % args.logfile
		exit()
	print 'TLC path = %s' % args.tlc
	print 'Socket = %s' % args.socket

	scheduler = Scheduler(args.socket, args.tlc)

	# switch to working directory
	os.chdir(args.dataset)

	logger = Logger(args.logfile)
	if 'Shared' in logger.headers:
		trainer = SharedTrainer(logger, args.dataset, scheduler)
	else:
		trainer = RegularTrainer(logger, args.dataset, scheduler)
	print
	print trainer.summary()
	
	trainer.train()