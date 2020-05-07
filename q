#!/bin/bash
#========================================
# Script to submit job in Intel devcloud
#
# Author: rakshith.krishnappa@intel.com
# Version: 0.4
#========================================
if [ -z "$1" ]; then
	echo "Missing script argument, Usage: ./q run.sh"
elif [ ! -f "$1" ]; then
    echo "File $1 does not exist"
else
	script=$1
	rm *.sh.* > /dev/null 2>&1
	#qsub
	echo "Submitting job:"
	qsub -l nodes=1:gpu:ppn=2 -d . $script
	#qstat
	qstat
	#wait for output file to be generated and display
	echo -ne "Waiting for Output."
	until [ -f $script.o* ]; do
		sleep 1
		echo -ne "."
		((timeout++))
		if [ $timeout == 40 ]; then
			echo "TimeOut 40 seconds: Job is queued, check for output file later (*.sh.o)"
			break
		fi
	done
	cat $script.o*
	cat $script.e*
fi
