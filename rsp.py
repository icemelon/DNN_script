from tokenparser import StringTokenParser

def removeItem(arr, filter):
	item = None
	for e in arr:
		if filter in e:
			item = e
			break
	if item is not None:
		arr.remove(item)

class RspDescription(object):
	def __init__(self, rspFile, idrop, hdrop, dropEpoch):
		self.firstEpoch = True
		self.drop = False
		self.idrop = idrop
		self.hdrop = hdrop
		self.dropEpoch = dropEpoch
		self.options = []
		self.classifier = {}

		self.parse(rspFile)

	def parseClassifier(self, description):
		detail = {}
		tokens = StringTokenParser(description.strip('{}'))

		while True:
			token = tokens.pop()
			if token is None: break
			if '=' in token:
				lhs, rhs = token.split('=')
				if lhs == 'bp':
					detail['bp'] = {}
					detail['bp']['name'] = rhs
					desc = tokens.pop()
					detail['bp']['detail'] = self.parseClassifier(desc)
				else:
					detail[lhs] = rhs
			else:
				detail[token] = None

		return detail

	def parse(self, rspFile):
		with open(rspFile) as f:
			for line in f:
				tokens = StringTokenParser(line)
				if tokens.pop() == '/cl':
					self.classifier['name'] = tokens.pop()
					self.classifier['detail'] = self.parseClassifier(tokens.pop())
				else:
					self.options.append(line.strip())
		# print self.classifier

	def removeFirstEpoch(self):
		if 'bp' in self.classifier['detail']:
			mcnn = self.classifier['detail']['bp']
		else:
			mcnn = self.classifier
		if 'initwts' in mcnn['detail']:
			del mcnn['detail']['initwts']
		if 'pretrain' in mcnn['detail']:
			del mcnn['detail']['pretrain']
		if 'prepoch' in mcnn['detail']:
			del mcnn['detail']['prepoch']
		self.firstEpoch = False

	def addDropout(self):
		if 'bp' in self.classifier['detail']:
			mcnn = self.classifier['detail']['bp']
		else:
			mcnn = self.classifier
		if self.idrop is not None:
			mcnn['detail']['idrop'] = self.idrop
		if self.hdrop is not None:
			mcnn['detail']['hdrop'] = self.hdrop
		self.drop = True

	# output /cl line
	def outputClassifier(self, lr, nn, cl=None):
		if cl is None:
			cl = self.classifier
		s = '%s { ' % cl['name']
		detail = cl['detail']

		for key in sorted(detail.keys()):
			if key == 'bp':
				value = self.outputClassifier(lr, nn, detail[key])
			else:
				value = detail[key]

			if value is None:
				format = '%s ' % key
			else:
				format = '%s=%s ' % (key, value)
			if key == 'lr':
				s += format % lr
			elif key == 'filename':
				s += format % nn
			else:
				s += format
		s += '}'
		return s

	def outputRspFile(self, task):
		if self.firstEpoch and task.epoch > 0:
			self.removeFirstEpoch()
		if not self.drop and task.epoch >= self.dropEpoch:
			self.addDropout()

		with open(task.rspFile, 'w') as f:
			for line in self.options:
				f.write('%s\n' % line)
			f.write('/cl %s\n' % self.outputClassifier(task.lr, task.nn))
			if task.rs is not None:
				f.write('/rs:%s\n' % task.rs)
			f.write('/m:%s\n' % task.binModel)
			f.write('/mt:%s\n' % task.textModel)

if __name__ == '__main__':
	import sys
	rsp = RspDescription(sys.argv[1], .1, .5, 0)
	# exit()
	class Task:
		pass
	task = Task()
	task.epoch = 1
	task.rs = 1
	task.nn = 'aaa.nn'
	task.lr = 0.1
	task.binModel = 'a'
	task.textModel = 'b'
	task.rspFile = '1'
	rsp.outputRspFile(task)