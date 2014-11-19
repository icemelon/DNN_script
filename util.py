import re

def parseValue(s):
	s = s.strip()
	if s == 'true':
		return True
	elif s == 'false':
		return False
	p = re.compile('^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')
	if p.match(s):
		return eval(s)

	raise Exception("%s is not a primitive value" % s)

def parseValueOrArray(s):
	s = s.strip(' []')
	if ',' in s:
		ret = []
		for data in s.split(','):
			ret.append(parseValue(data))
		return ret
	else:
		return parseValue(s)
