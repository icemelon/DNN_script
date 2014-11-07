import re

from const import Const

def parseLine(line):
	line = line.strip()
	cmd = line[1:line.index(']')]
	content = line[line.index(']')+1:].strip()
	return (cmd, content)

def parseContent(content):
	entry = {}
	p = re.compile('\d+(\.\d+)?')
	for e in content.split():
		data = e.split('=')
		if data[1].startswith('['):
			l = []
			for e in data[1].strip('[]').split(','):
				if p.match(e):
					e = eval(e)
				l.append(e)
			data[1] = l
		elif p.match(data[1]):
			data[1] = eval(data[1])
		entry[data[0]] = data[1]
	return entry

class Logger(object):
	def __init__(self, logfile):
		self.logfile = logfile
		self.headers = {}
		self.history = []
		self.parse()

	def parse(self):
		with open(self.logfile) as f:
			header = f.readline()
			logs = f.readlines()
		self.parseHeader(header)
		self.parseHistory(logs)

	def log(self, content):
		with open(self.logfile, 'a') as out:
			out.write('%s\n' % content)
			out.flush()

	def parseHeader(self, header):
		for item in header.split():
			data = item.split('=')
			data[0] = data[0].lower()
			if data[0] == 'name':
				self.headers['ThreadName'] = data[1]
			elif data[0] == 'rsptemplate':
				# load template rsp file
				self.headers['RspTmplFile'] = data[1]
			elif data[0] == 'lr':
				self.headers['lr'] = eval(data[1])
			elif data[0] == 'nn':
				self.headers['nn'] = data[1]
			elif data[0] == 'lred':
				self.headers['lred'] = eval(data[1])
			elif data[0] == 'rs':
				self.headers['rs'] = Const.parseValueOrArray(data[1])
			elif data[0] == 'idrop':
				self.headers['idrop'] = eval(data[1])
			elif data[0] == 'hdrop':
				self.headers['hdrop'] = eval(data[1])
			elif data[0] == 'dropepoch':
				self.headers['DropEpoch'] = eval(data[1])
			elif data[0] == 'trainednn':
				self.headers['TrainedNN'] = data[1]
			elif data[0] == 'shared':
				self.headers['Shared'] = eval(data[1])

	def parseHistory(self, logs):
		index = 0
		while index < len(logs):
			cmd, content = parseLine(logs[index])
			index += 1

			if cmd == 'JOB NEW':
				job = parseContent(content)
				job['Fork'] = False
				job['Finish'] = False
				if len(self.history) > 0 and job['Epoch'] == self.history[-1]['Epoch']:
					self.history[-1] = job
				else:
					self.history.append(job)

			elif cmd == 'JOB END':
				entry = parseContent(content)
				job = self.history[-1]
				job.update(entry)
				job['Finish'] = True

			elif cmd == 'FORKJOB NEW':
				job = parseContent(content)
				job['Fork'] = True
				job['TaskList'] = {}
				job['Finish'] = False
				if len(self.history) > 0 and job['Epoch'] == self.history[-1]['Epoch']:
					self.history[-1] = job
				else:
					self.history.append(job)

			elif cmd == 'FORKJOB END':
				entry = parseContent(content)
				job = self.history[-1]
				job.update(entry)
				job['Finish'] = True

			elif cmd == 'TASK NEW':
				task = parseContent(content)
				task['Finish'] = False
				self.history[-1]['TaskList'][task['SubId']] = task

			elif cmd == 'TASK END':
				entry = parseContent(content)
				entry['Finish'] = True
				self.history[-1]['TaskList'][entry['SubId']].update(entry)

			elif cmd == 'UPDATE':
				# no longer support UPDATE
				continue
