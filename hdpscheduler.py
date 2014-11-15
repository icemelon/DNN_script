import os
import time
import json
import argparse
from lockfile import LockFile

from config import *
import hdp
import run

ROOTDIR = os.path.dirname(__file__)
INFO_DIR = os.path.join(ROOTDIR, 'TaskInfo')
LOCK_FILE = os.path.join(INFO_DIR, '.lock')
PENDING_FILE = os.path.join(INFO_DIR, 'pending.txt')
ACTIVE_FILE = os.path.join(INFO_DIR, 'active.txt')
FINISH_FILE = os.path.join(INFO_DIR, 'finish.txt')

def loadTasks(filename):
	lock = LockFile(LOCK_FILE)
	lock.acquire()
	with open(filename) as f:
		content = f.read()
	if len(content.strip()) == 0:
		ret = []
	else:
		ret = json.loads(content)
	lock.release()
	return ret

def dumpTasks(filename, tasklist):
	lock = LockFile(LOCK_FILE)
	lock.acquire()
	with open(filename, 'w') as f:
		f.write("[\n  ")
		f.write(",\n  ".join(json.dumps(task) for task in tasklist))
		f.write("\n]\n")
	lock.release()

def add(args):
	task = {}
	for var in args.__dict__:
		if var == 'func': continue
		task[var] = args.__dict__[var]

	tasklist = loadTasks(PENDING_FILE)
	tasklist.append(task)
	dumpTasks(PENDING_FILE, tasklist)

def status(args):
	pendingTasks = loadTasks(PENDING_FILE)
	activeTasks = loadTasks(ACTIVE_FILE)
	finishTasks = loadTasks(FINISH_FILE)
	print 'Pending Tasks:'
	for task in pendingTasks:
		print task
	print
	print 'Active Tasks:'
	for task in activeTasks:
		print task
	print
	print 'Finished Tasks:'
	for task in finishTasks:
		print task

def checkFinish(activeTasks, finishTasks):
	finished = []
	for task in activeTasks:
		if hdp.finish(task['jobName']):
			with open(task['logfile']) as f:
				log = f.read()
			if '[FINISH]' in log:
				finished.append(task)
			else:
				# restart the job
				task['jobName'] = run.runHDP(task)
				print "======================================"

	finishTasks.extend(finished)
	for task in finished:
		activeTasks.remove(task)

	if len(finished) > 0:
		dumpTasks(ACTIVE_FILE, activeTasks)
		dumpTasks(FINISH_FILE, finishTasks)

def daemon(args):
	activeTasks = loadTasks(ACTIVE_FILE)
	finishTasks = loadTasks(FINISH_FILE)

	checkFinish(activeTasks, finishTasks)

	while True:
		pendingTasks = loadTasks(PENDING_FILE)
		if len(pendingTasks) > 0:
			for task in pendingTasks:
				task['jobName'] = run.runHDP(task)
				print "======================================"
				activeTasks.append(task)
			pendingTasks = []
			dumpTasks(PENDING_FILE, pendingTasks)
			dumpTasks(ACTIVE_FILE, activeTasks)

		checkFinish(activeTasks, finishTasks)
		time.sleep(120) # check every 2 minutes

if __name__ == '__main__':
	mainParser = argparse.ArgumentParser(description='HDP Job Scheduler')
	subparsers = mainParser.add_subparsers(help='command help')

	addParser = subparsers.add_parser('add', help='Add one training task')
	addParser.set_defaults(func=add)
	addParser.add_argument('dataset', help='Working directory that contains dataset')
	addParser.add_argument('logfile', help='Log file (header should be initialed)')
	addParser.add_argument('-s', '--socket', type=int, default=1, help='Available socket number (Default: 1)')
	addParser.add_argument('--tlc', default=DEFAULT_HDP_TLC_PATH, help='TLC executable path')
	addParser.add_argument('--jobtemp', choices=['ExpressQ', 'Cuda65-Nodes', 'DevNodes'], default='ExpressQ', help='Select job template')
	addParser.add_argument('--priority', choices=['High', 'Normal', 'Low'], default='Normal', help='Job priority')

	statusParser = subparsers.add_parser('status', help='Check the status')
	statusParser.set_defaults(func=status)

	daemonParser = subparsers.add_parser('daemon', help='Start the daemon')
	daemonParser.set_defaults(func=daemon)

	try:
		args = mainParser.parse_args()
	except:
		exit()

	if not os.path.exists(INFO_DIR):
		os.mkdir(INFO_DIR)
	if not os.path.exists(PENDING_FILE):
		with open(PENDING_FILE, 'w') as f:
			f.write("")
	if not os.path.exists(ACTIVE_FILE):
		with open(ACTIVE_FILE, 'w') as f:
			f.write("")
	if not os.path.exists(FINISH_FILE):
		with open(FINISH_FILE, 'w') as f:
			f.write("")

	args.func(args)
	# run.parser
