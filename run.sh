#!/bin/bash
echo "Starting server"
python server/main.py & sleep 3

intexit() {
    echo -e "Killing everything"
    trap - SIGINT # restore default SIGINT handler
    kill -9 $(pgrep python)
    exit
}

trap intexit SIGINT

echo "Starting the clients"
for i in {0..4..1} 
do
    echo "Starting client $i"
    python client/run_client.py --cid=${i} &
done
wait