#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
echo "########## Executing the run"
#./bin/benchOneAPI
./bin/cdot
./bin/mvec
./bin/matmat
echo "########## Done with the run"
