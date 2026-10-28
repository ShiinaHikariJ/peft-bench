#!/bin/bash
while ! grep -q QUEUE_DONE /root/autodl-tmp/bench/main.log; do sleep 60; done
cd /root/autodl-tmp/bench && ./queue.sh > main2.log 2>&1
