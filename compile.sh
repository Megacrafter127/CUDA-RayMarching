#!/bin/bash
command='nvcc -O3 -cudart static --relocatable-device-code=true -o RayMarching -I"headers/" src/* -lm -lSDL2'
echo "$command"
$command
