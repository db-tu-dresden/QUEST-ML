#!/usr/bin/env bash

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

#COMMAND=python3 create_data.py --path './save/-/'

printf "%s\n\n" "$COMMAND" | tee $RESULTS

for i in {0..10};
do
  { time "$COMMAND"; } 2>&1 | grep real | tee $RESULTS
done