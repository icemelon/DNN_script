import os

from config import TMP_DIR

class Task(object):
	def __init__(self, rspTmpl, rootdir, threadName, epoch, lr, nn, rs, subId):
		self.rootdir = rootdir
		self.threadName = threadName
		self.nn = nn
		self.epoch = epoch
		self.lr = lr
		self.rs = rs
		self.subId = subId

		if subId == 0:
			idName = '%s.%s' % (threadName, epoch)
		else:
			idName = '%s.%s.%s' % (threadName, epoch, subId)

		self.taskName = '%s_%s' % (os.path.basename(rootdir), idName)
		self.rspFile = os.path.join(TMP_DIR, '%s.rsp' % idName)
		self.stdout = os.path.join(TMP_DIR, '%s.Result.txt' % idName)
		self.rawoutput = os.path.join(TMP_DIR, '%s.raw.txt' % idName)
		self.binModel = os.path.join('model', '%s.Model.bin' % idName)
		self.textModel = os.path.join('model', '%s.Model.nn' % idName)
		#self.stderr = '%s.Error.txt' % idName

		# create rsp File
		rspTmpl.outputRspFile(self)

class ForkTask(object):
	def __init__(self, rspTmpl, rootdir, threadName, epoch, lr, nn, rsList):
		self.taskName = '%s_%s.%s' % (os.path.basename(rootdir), threadName, epoch)
		self.taskList = []
		self.taskCount = len(rsList)
		for subId in range(self.taskCount):
			self.taskList.append(Task(rspTmpl, rootdir, threadName, epoch, lr, nn, rsList[subId], subId))