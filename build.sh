#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh #> setvars.log
dpcpp src/benchOneAPI.cpp -o bin/benchOneAPI -std=c++11 -fsycl

