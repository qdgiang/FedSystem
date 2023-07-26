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
wait