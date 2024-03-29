import subprocess

from config import *

# (Job Template -> Runtime)
HDP_JOBTEMP_RUNTIME = {
	'ExpressQ': "0:12:0", # upto 12 hours
	'Cuda65-Nodes': "14:0:0", # upto 2 weeks 
	'DevNodes': "0:4:0", # upto 4 hours
}

def create(jobTemp, jobName, nodes):
	cmd = 'job new /scheduler:%s /jobtemplate:%s /numnodes:%s-%s /jobname:%s /runtime:%s' \
		% (DEFAULT_HDP_SCHEDULER, jobTemp, nodes, nodes, jobName, HDP_JOBTEMP_RUNTIME[jobTemp])
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	ret = p.stdout.read().split('\n')
	jobID = None
	for line in ret:
		if line.startswith('Created'):
			jobID = eval(line[line.index('ID:')+3:])
			break
	if (jobID == None):
		print 'Error in creating a new job'
		print '\n'.join(ret)
	return jobID

def addTask(jobID, nodes, workdir, command, stdout, stderr):
	cmd = 'job add %s /scheduler:%s /workdir:%s /numnodes:%s-%s /stdout:%s /stderr:%s %s' \
		% (jobID, DEFAULT_HDP_SCHEDULER, workdir, nodes, nodes, stdout, stderr, command)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	ret = p.stdout.read()
	if 'added' in ret:
		return True
	else:
		print ret
		return False

def submit(jobID):
	cmd = 'job submit /scheduler:%s /id:%s' % (DEFAULT_HDP_SCHEDULER, jobID)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	ret = p.stdout.read()
	if 'submitted' in ret:
		return True
	else:
		print ret
		return False

def execute(jobTemp, jobName, nodes, workdir, command, stdout, stderr):
	jobID = create(jobTemp, jobName, nodes)
	if jobID is None:
		return False
	print "Created job %s" % jobID

	if not addTask(jobID, nodes, workdir, command, stdout, stderr):
		return False
	print "Added task \"%s\" (stdout: %s, stderr: %s)" % (command, stdout, stderr)
	
	if not submit(jobID):
		return False
	print "Submitted job %s" % jobID

def finish(jobName):
	cmd = 'job list /scheduler:%s /jobname:%s' % (DEFAULT_HDP_SCHEDULER, jobName)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	ret = p.stdout.read()
	if 'Running' in ret or 'Queued' in ret:
		return False
	else:
		return True
