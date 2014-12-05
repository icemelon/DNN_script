import util
from tokenparser import *
from bundle import *

DefaultHiddenLayerFunction = 'sigmoid'
DefaultOutputLayerFunction = 'softmax'

class Layer(object):
	def __init__(self, name, dimOutput, params):
		self.name = name
		self.params = params
		self.dimOutput = dimOutput

	def removeParams(self):
		pass

	# output to string
	def output(self):
		pass

	def numParam(self):
		pass

	def numComput(self):
		pass

	def numOutput(self):
		if type(self.dimOutput) is list:
			return reduce(lambda x,y: x*y, self.dimOutput)
		else:
			return self.dimOutput

	# def computeDimOutput(self):
	# 	return self.dimOutput

	def _dimOutput_to_str(self):
		if type(self.dimOutput) is list:
			return str(self.dimOutput)
		else:
			return "[%s]" % self.dimOutput

	@staticmethod
	def parse(tokens, net):
		layerType = tokens.pop()
		name = tokens.pop()
		dimOutput = util.parseValueOrArray(tokens.pop())

		if layerType == 'input':
			return InputLayer(name, dimOutput, net.params)

		token = tokens.pop()
		if 'from' not in token:
			outputFunc = token
			token = tokens.pop()
		else:
			outputFunc = None

		attrs = {}
		if token[0] == '{':
			tokens = StringTokenParser(token.strip('{}'))
			while not tokens.startswith("from"):
				desc = tokens.pop(separators=";")
				key, val = desc.split('=')
				attrs[key.strip()] = net.params.parseParam(val)
			tokens.pop() # skip "from"
		else:
			assert token == "from"

		inputLayerName = tokens.pop()
		inputLayer = net.findLayerByName(inputLayerName)
		# print "input: %s" % inputLayer
		if inputLayer is None:
			raise Exception('Cannot find input layer: %s' % inputLayerName)

		bundle = Bundle.parse(inputLayer, tokens, net.params)

		if layerType == "hidden":
			return HiddenLayer(name, dimOutput, net.params, outputFunc, bundle, attrs)
		elif layerType == "output":
			return OutputLayer(name, dimOutput, net.params, outputFunc, bundle, attrs)
		else:
			raise Exception('Wrong layer type: %s' % layerType)

class InputLayer(Layer):
	def __init__(self, name, dimOutput, params):
		super(InputLayer, self).__init__(name, dimOutput, params)

	def numParam(self):
		return 0

	def numComput(self):
		return 0

	def output(self):
		return 'input %s %s;' % (self.name, self._dimOutput_to_str())

class HiddenLayer(Layer):
	def __init__(self, name, dimOutput, params, outputFunc, bundle, attrs):
		super(HiddenLayer, self).__init__(name, dimOutput, params)
		# output function can be rlinear, sigmoid, softmax
		self.outputFunc = DefaultHiddenLayerFunction if outputFunc is None else outputFunc
		# bundle is the input function
		# could be convolve, fully, max pool, responive norm
		self.bundle = bundle
		self.biases = attrs.get("Biases", None)

	def removeParams(self):
		if self.biases is not None:
			self.params.remove(self.biases)
			self.biases = None
		self.bundle.removeParams()

	def numParam(self):
		if type(self.bundle) is FullBundle:
			return self.numOutput() * (self.bundle.input.numOutput() + 1)
		elif type(self.bundle) is ConvolveBundle:
			# print self.bundle.sharing
			sharing = True
			if self.bundle.sharing is not None:
				sharing = self.bundle.sharing[-1]
			if sharing:
				# global convolution
				if type(self.bundle.mapCount) is list:
					mapCount = self.bundle.mapCount[0]
				else:
					mapCount = self.bundle.mapCount
				return (self.bundle.geo.numKernel() + 1) * mapCount
			else:
				# local convolution
				return (self.bundle.geo.numKernel() + 1) * self.numOutput()
		else:
			return 0

	def numComput(self):
		if type(self.bundle) is FullBundle:
			ops = 2 * self.bundle.input.numOutput() * self.numOutput()
		elif type(self.bundle) is ConvolveBundle:
			ops = 2 * self.bundle.geo.numKernel() * self.numOutput()
		else:
			ops = 0
		return ops

	def output(self):
		s = "hidden %s %s %s {\n" % (self.name, self._dimOutput_to_str(), self.outputFunc)
		if self.biases is not None:
			s += "  Biases = %s;\n" % self.params.output(self.biases)
		s += "%s\n}" % self.bundle.output()
		return s

class OutputLayer(Layer):
	def __init__(self, name, dimOutput, params, outputFunc, bundle, attrs):
		super(OutputLayer, self).__init__(name, dimOutput, params)
		self.outputFunc = DefaultHiddenLayerFunction if outputFunc is None else outputFunc
		self.bundle = bundle
		self.biases = attrs.get("Biases", None)

	def removeParams(self):
		if self.biases is not None:
			self.params.remove(self.biases)
			self.biases = None
		self.bundle.removeParams()

	def numParam(self):
		if type(self.bundle) is FullBundle:
			return self.numOutput() * (self.bundle.input.numOutput() + 1)
		elif type(self.bundle) is ConvolveBundle:
			if self.bundle.sharing is None or self.bundle.sharing[-1]:
				# global convolution
				if type(self.bundle.mapCount) is list:
					mapCount = self.bundle.mapCount[0]
				else:
					mapCount = self.bundle.mapCount
				return (self.bundle.geo.numKernel() + 1) * mapCount
			else:
				# local convolution
				return (self.bundle.geo.numKernel() + 1) * self.numOutput()
		else:
			return 0

	def numComput(self):
		if type(self.bundle) is FullBundle:
			ops = 2 * self.bundle.input.numOutput() * self.numOutput()
		elif type(self.bundle) is ConvolveBundle:
			ops = 2 * self.bundle.geo.numKernel() * self.numOutput()
		else:
			ops = 0
		return ops

	def output(self):
		s = "output %s %s %s {\n" % (self.name, self._dimOutput_to_str(), self.outputFunc)
		if self.biases is not None:
			s += "  Biases = %s;\n" % self.params.output(self.biases)
		s += "%s\n}" % self.bundle.output()
		return s

