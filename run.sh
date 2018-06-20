#!/bin/bash
bin=~/.conda/envs/mypython/bin/
app=~/project/Jigsaw
if [ -f out ]; then
rm -f out
echo "rm -r out"
fi
$bin/python $app/game.py $1 1>>out 2>&1 & 
