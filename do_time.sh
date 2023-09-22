#!/usr/bin/env bash

trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

while getopts ":n:c:" opt; do
  case $opt in
    n) N="$OPTARG"
    ;;
    c) COMMAND="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done


RESULTS=./_time_results.txt
touch $RESULTS
truncate -s 0  $RESULTS

echo "Running $N times: $COMMAND" | tee $RESULTS

for i in $(seq 1 $N);
do
  { time eval $COMMAND; } 2>&1 | grep real | awk '{print $2}' | tee -a $RESULTS
done