import os
import sys
import time
import subprocess
import argparse

from task import *
from config import *
from logger import Logger
from trainer import Trainer

class Scheduler(object):
	def __init__(self, maxSocket):
		self.maxSocket = maxSocket
		self.pending = []
		self.running = {}

	def log(self, msg):
		sys.stdout.write("%s\n" % msg)
		sys.stdout.flush()

	def updateAndRun(self):
		finished = []
		for task in self.running:
			proc = self.running[task]
			if proc.poll() is not None:
				self.log("[%s] Task %s finished" % (time.asctime(), task.taskName))
				finished.append(task)

		for task in finished:
			del self.running[task]

		while len(self.pending) > 0 and len(self.running) < self.maxSocket:
			task = self.pending.pop(0)
			cmd = "%s @%s > %s" % (TLC_PATH, task.rspFile, task.stdout)
			proc = subprocess.Popen(cmd, cwd=task.rootdir, shell=True, universal_newlines=True)
			self.running[task] = proc
			self.log("[%s] Task %s is running" % (time.asctime(), task.taskName))

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
	parser.add_argument('socket', type=int, nargs='?', default=1, help='Socket number to use in the HDP job (Default: 1)')

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
	print 'Socket = %s' % args.socket

	scheduler = Scheduler(args.socket)

	logger = Logger(args.logfile)
	trainer = Trainer(logger, args.dataset, scheduler)
	print trainer.summary()
	
	trainer.train()