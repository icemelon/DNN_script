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

	# fin for the rest of input file
	@staticmethod
	def output(fout, fin=None):
		for key in Const.values:
			val = Const.values[key]
			if type(val) is list:
				fout.write("const %s = %s;\n" % (key, val))

		if fin is not None:
			while True:
				line = fin.readline()
				if not line: break
				line = line.strip('\n')
				if len(line) == 0: continue
				if not line.startswith("const"): continue

				var = line.split()[1]
				# check if this const is required
				if var in Const.values:
					fout.write("\n%s\n" % line)
					end = False
					while not end:
						line = fin.readline().strip('\n')
						fout.write("%s\n" % line)
						if line.endswith(";"): end = True
				else:
					# skip this const
					end = False
					while not end:
						line = fin.readline().strip('\n')
						if line.endswith(";"): end = True

