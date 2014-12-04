from tokenparser import StringTokenParser

class RspTemplate(object):
	def __init__(self):
		self.core = None
		self.classifier = None
		self.trainDataset = None
		self.testDataset = None
		self.rs = None
		self.options = {}

	@staticmethod
	def parseRspFile(rspTmplFile):
		content = ""
		with open(rspTmplFile) as f:
			for line in f:
				content += " " + line.strip()
		tokens = StringTokenParser(content)
		rspTmpl = RspTemplate()
		while True:
			token = tokens.pop(': ')
			if token is None:
				break
			elif token == '/c':
				rspTmpl.core = tokens.pop()
				rspTmpl.trainDataset = tokens.pop()
			elif token == '/test':
				rspTmpl.testDataset = tokens.pop()
			elif token == '/rs':
				rspTmpl.rs = tokens.pop()
			elif token == '/cl':
				alg = tokens.pop()
				options = RspTemplate.parseClassifier(tokens.pop())
				rspTmpl.classifier = {'name': alg, 'options': options}
			else:
				if token.endswith('-'):
					rspTmpl.options[token[:-1]] = '-'
				else:
					rspTmpl.options[token] = tokens.pop()
		return rspTmpl

	@staticmethod
	def parseClassifier(description):
		options = {}
		tokens = StringTokenParser(description.strip('{}'))

		while True:
			token = tokens.pop()
			if token is None: break
			if '=' in token:
				lhs, rhs = token.split('=')
				if lhs == 'bp':
					alg = rhs
					opt = RspTemplate.parseClassifier(tokens.pop())
					options['bp'] = {'name': alg, 'options': opt}
				else:
					options[lhs] = rhs
			else:
				options[token] = None

		return options

	def removeFirstEpoch(self):
		if 'bp' in self.classifier['options']:
			cl_options = self.classifier['options']['bp']['options']
		else:
			cl_options = self.classifier['options']
		if 'initwts' in cl_options:
			del cl_options['initwts']
		if 'pretrain' in cl_options:
			del cl_options['pretrain']
		if 'prepoch' in cl_options:
			del cl_options['prepoch']

	def addDropout(self, idrop, hdrop):
		if 'bp' in self.classifier['options']:
			cl_options = self.classifier['options']['bp']['options']
		else:
			cl_options = self.classifier['options']
		if idrop is not None:
			cl_options['idrop'] = idrop
		if hdrop is not None:
			cl_options['hdrop'] = hdrop

	# output /cl line
	def outputClassifier(self, lr, nn, cl=None):
		if cl is None:
			cl = self.classifier
		s = '%s { ' % cl['name']
		options = cl['options']
		for key in sorted(options.keys()):
			if key == 'bp':
				value = self.outputClassifier(lr, nn, options[key])
			elif key == 'lr':
				value = lr
			elif key == 'filename':
				value = nn
			else:
				value = options[key]

			if value is None:
				s += '%s ' % key
			else:
				s += '%s=%s ' % (key, value)
		s += '}'
		return s

	def writeToFile(self, filename, lr, nn, rs, binModel, textModel, rawoutput):
		with open(filename, 'w') as f:
			f.write("/c %s %s\n" % (self.core, self.trainDataset))
			if self.testDataset is not None:
				f.write("/test %s\n" % self.testDataset)
			for key in self.options:
				val = self.options[key]
				if val == '-':
					f.write("%s%s\n" % (key, self.options[key]))
				else:
					f.write("%s %s\n" % (key, self.options[key]))
			f.write("/cl %s\n" % self.outputClassifier(lr, nn))
			if rs is None and self.rs is not None: rs = self.rs
			if rs is not None: f.write("/rs %s\n" % rs)
			f.write("/m %s\n" % binModel)
			if textModel is not None: f.write("/mt %s\n" % textModel)
			f.write("/raw %s\n" % rawoutput)

class RspGenerator(object):
	# rspTmpl could be either RspTemplate obj, or rsp template file
	def __init__(self, rspTmpl, idrop, hdrop, dropEpoch):
		if type(rspTmpl) is RspTemplate:
			self.rspTmpl = rspTmpl
		else:
			self.rspTmpl = RspTemplate.parseRspFile(rspTmpl)
		self.idrop = idrop
		self.hdrop = hdrop
		self.dropEpoch = dropEpoch

	def outputRspFile(self, task):
		if task.epoch > 0:
			self.rspTmpl.removeFirstEpoch()
		if task.epoch >= self.dropEpoch:
			self.rspTmpl.addDropout(self.idrop, self.hdrop)
		self.rspTmpl.writeToFile(
			task.rspFile, 
			task.lr, 
			task.nn, 
			task.rs, 
			task.binModel, 
			task.textModel,
			task.rawoutput)

if __name__ == '__main__':
	import sys
	# rspTmpl = RspTemplate.parseRspFile(sys.argv[1])
	# rspTmpl.writeToFile("1.rsp", 0.1, 'a.nn', 1, 'b.bin', 'b.nn')
	# exit()
	rsp = RspGenerator(sys.argv[1], .1, .5, 0)
	# exit()
	class Task:
		pass
	task = Task()
	task.epoch = 1
	task.rs = 1
	task.nn = 'a0.nn'
	task.lr = 0.1
	task.binModel = 'a1.bin'
	task.textModel = 'a1.nn'
	task.rspFile = '1.rsp'
	rsp.outputRspFile(task)