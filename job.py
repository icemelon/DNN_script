import sys
import time
import subprocess

from task import *

class JobManager(object):
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
