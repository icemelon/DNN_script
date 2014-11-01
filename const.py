import re

class Const(object):
	values = {}
	
	@staticmethod
	def parse(tokens):
		tokens.pop() # skip "const"
		if tokens.content.startswith('{'):
			desc = tokens.pop().strip('{}')
			for item in desc.split(';'):
				item = item.strip()
				if len(item) == 0: continue
				key, val = item.split('=')
				Const.values[key.strip()] = Const.parseValueOrArray(val)
		else:
			item = tokens.pop(sep=";")
			key, val = item.split('=')
			Const.values[key.strip()] = Const.parseValueOrArray(val)

	@staticmethod
	def parseValue(s):
		s = s.strip()
		if s == 'true':
			return True
		elif s == 'false':
			return False
		p = re.compile('^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')
		if p.match(s):
			return eval(s)
		if s in Const.values:
			val = Const.values[s]
			if type(val) is list:
				# list type only use symbols
				return s
			else:
				return Const.values[s]
		else:
			# let's put dangling const first
			Const.values[s] = None
			return s
		# err = 'Value "%s" cannot be parsed' % s
		#raise Exception(err)

	@staticmethod
	def parseValueOrArray(s):
		s = s.strip(' []')
		if ',' in s:
			ret = []
			for data in s.split(','):
				ret.append(Const.parseValue(data))
			return ret
		else:
			return Const.parseValue(s)

	@staticmethod
	def tostr(val):
		if type(val) is bool:
			return 'true' if val else 'false'
		elif type(val) is list:
			return '[' + ', '.join(Const.tostr(x) for x in val) + ']'
		else:
			return str(val)

	@staticmethod
	def output(out):
		for key in Const.values:
			val = Const.values[key]
			if type(val) is list:
				out.write("const %s = %s;\n" % (key, val))
