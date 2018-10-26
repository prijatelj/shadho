#!/usr/bin/env bash

ps aux | grep $1 | grep work_queue | awk '{print $2}' | xargs -L1 kill
ps aux | grep $1 | grep shadho_wq | awk '{print $2}' | xargs -L1 kill
