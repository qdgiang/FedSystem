#!/bin/bash
intexit() {
    echo -e "Killing everything"
    trap - SIGINT # restore default SIGINT handler
    kill -9 $(pgrep python)
    exit
}

trap intexit SIGINT

echo "Starting client"
for i in {0..5..1} 
do
    echo "Starting client $i"
    python client/run_client.py --cid=${i} &
done
wait