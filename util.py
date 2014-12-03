import re
import struct

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

def parseLargeArray(s):
	s = s.strip(' []')
	ret = []
	for data in s.split(','):
		ret.append(eval(data))
	return ret

def parseHex(s):
	return struct.unpack('<f', s.decode('hex'))[0]