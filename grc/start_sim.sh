#!/bin/bash

# Quick workaround: be sure to set the correct path to OPINCAA
OPINCAA_PATH=../../opincaa

SIM_PATH=$OPINCAA_PATH/simulator

FIFO_ROOT_PATH=$PWD

DISTRIBUTION_FIFO_SIM=$FIFO_ROOT_PATH/distributionFIFO
REDUCTION_FIFO_SIM=$FIFO_ROOT_PATH/reductionFIFO
WRITE_FIFO_SIM=$FIFO_ROOT_PATH/writeFIFO
READ_FIFO_SIM=$FIFO_ROOT_PATH/readFIFO
    
    
#start the simulator
./$SIM_PATH/build/simulator

#run the test
sleep 3
