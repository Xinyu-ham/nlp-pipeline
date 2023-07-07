#!/bin/bash
for subdir in ./kubernetes/*
do
    for file in $subdir/*
    do
        if [[ $file == *.template ]]
        then
            new_file=${file:0:${#file} - 9}
            envsubst < $file > $new_file
        fi
    done
done