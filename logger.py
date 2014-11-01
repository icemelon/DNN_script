class Logger(object):
	def __init__(self, logfile):
		self.logfile = logfile

	def readToEnd(self):
		with open(self.logfile) as f:
			header = f.readline()
			logs = f.readlines()
		return header, logs

	def log(self, content):
		with open(self.logfile, 'a') as out:
			out.write('%s\n' % content)
			out.flush()
