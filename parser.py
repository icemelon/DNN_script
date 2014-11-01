import os
import re
import sys
import argparse

from nn import NeuralNetwork
from const import Const

def isnum(expr):
	p = re.compile('\d+(\.\d+)?')
	if p.match(expr):
		return True
	else:
		return False

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(description='Parse/Convert TLC NN file')
	argparser.add_argument('input', help='input file name (.nn)')
	# argparser.add_argument('-v', '--verify', action='store_true', dest='verify', help='Verify NN definition')
	argparser.add_argument('-m', '--modify', help='Modification description file')
	argparser.add_argument('-o', '--output', help='output file name (.nn)')
	try:
		args = argparser.parse_args()
	except:
		exit()
	
	nn = NeuralNetwork.parseNN(args.input)
	# if args.verify:
	# 	if nn.verify():
	# 		print '%s is legal' % args.input
	# 	else:
	# 		print '%s is not legal' % args.input
	# 	exit()
	
	if args.modify is not None:
		with open(args.modify) as f:
			content = f.readlines()
		i = 0
		while i < len(content):
			line = content[i].strip()
			i += 1
			tokens = line.split(':')
			assert(tokens[0] == 'Layer')
			layerName = tokens[1].strip()
			mod = {}
			while i < len(content):
				line = content[i].strip()
				if line.startswith('Layer'):
					break
				i += 1
				tokens = line.split(':')
				expr = tokens[1].strip()
				if isnum(expr):
					op = '='
					val = eval(expr)
				else:
					op = expr[0]
					val = eval(expr[1:])
				mod[tokens[0].strip()] = (val, op)
			nn.update(layerName, mod)
		# if not nn.verify():
		# 	exit()

	if args.output is None:
		print nn
		Const.output(sys.stdout)
	else:
		with open(args.output, 'w') as out:
			out.write(str(nn))
			Const.output(out)
