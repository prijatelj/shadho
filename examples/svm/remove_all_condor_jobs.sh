#!/usr/bin/env bash

condor_q $1 | awk '{print $1}' | grep '\.' | xargs -L1 condor_rm
