#!/usr/bin/env python

import subprocess


p = subprocess.Popen("python test11.py", shell=True, stdout=subprocess.PIPE)


# Execute your training algorithm.
def _run(cmd):
    """Invokes your training algorithm."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
    stdout, stderr = process.communicate()
    return_code = process.poll()

    while return_code is None:
            output = process.stdout.readline()
            print (output.strip())

