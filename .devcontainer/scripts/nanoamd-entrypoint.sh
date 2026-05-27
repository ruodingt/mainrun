#!/bin/bash
tensorboard --logdir /home/ubuntu/workspace/mainrun/experiments --host=0.0.0.0 --port=6006 &
exec sudo /usr/sbin/sshd -D
