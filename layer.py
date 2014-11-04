from tokenparser import *
from const import Const
from bundle import *

DefaultHiddenLayerFunction = 'sigmoid'
DefaultOutputLayerFunction = 'softmax'

class Layer(object):
	def __init__(self, name, dimOutput):
		self.name = name
		self.dimOutput = dimOutput

	def computeDimOutput(self):
		return self.dimOutput

	def dimOutput_to_str(self):
		if type(self.dimOutput) is list:
			return str(self.dimOutput)
		else:
			return "[%s]" % self.dimOutput

	@staticmethod
	def parse(tokens, nn):
		layerType = tokens.pop()
		name = tokens.pop()
		dimOutput = Const.parseValueOrArray(tokens.pop())

		if layerType == 'input':
			return InputLayer(name, dimOutput)

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
				desc = tokens.pop(seperators=";")
				key, val = desc.split('=')
				attrs[key.strip()] = Const.parseValueOrArray(val)
			tokens.pop() # skip "from"
		else:
			assert token == "from"

		inputLayerName = tokens.pop()
		inputLayer = nn.findLayerByName(inputLayerName)
		# print "input: %s" % inputLayer
		if inputLayer is None:
			raise Exception('Cannot find input layer: %s' % inputLayerName)

		bundle = Bundle.parse(inputLayer, tokens)

		if layerType == "hidden":
			return HiddenLayer(name, dimOutput, outputFunc, bundle, attrs)
		elif layerType == "output":
			return OutputLayer(name, dimOutput, outputFunc, bundle, attrs)
		else:
			raise Exception('Wrong layer type: %s' % layerType)

class InputLayer(Layer):
	def __init__(self, name, dimOutput):
		super(InputLayer, self).__init__(name, dimOutput)

	def __str__(self):
		return 'input %s %s;' % (self.name, self.dimOutput_to_str())

class HiddenLayer(Layer):
	def __init__(self, name, dimOutput, outputFunc, bundle, attrs):
		super(HiddenLayer, self).__init__(name, dimOutput)
		# output function can be rlinear, sigmoid, softmax
		self.outputFunc = DefaultHiddenLayerFunction if outputFunc is None else outputFunc
		# bundle is the input function
		# could be convolve, fully, max pool, responive norm
		self.bundle = bundle
		self.biases = attrs.get("Biases", None)

	def __str__(self):
		s = "hidden %s %s %s {\n" % (self.name, self.dimOutput_to_str(), self.outputFunc)
		if self.biases is not None:
			s += "  Biases = %s;\n" % Const.tostr(self.biases)
		s += "%s\n}" % self.bundle
		return s

class OutputLayer(Layer):
	def __init__(self, name, dimOutput, outputFunc, bundle, attrs):
		super(OutputLayer, self).__init__(name, dimOutput)
		self.outputFunc = DefaultHiddenLayerFunction if outputFunc is None else outputFunc
		self.bundle = bundle
		self.biases = attrs.get("Biases", None)

	def __str__(self):
		s = "output %s %s %s" % (self.name, self.dimOutput_to_str(), self.outputFunc)
		if self.biases is None:
			s += "\n%s" % self.bundle
		else:
			s += " {\n  Biases = %s;\n" % Const.tostr(self.biases)
			s += "%s\n}" % self.bundle
		return s