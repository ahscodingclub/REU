#!/bin/bash
nparent=( 10 20 )
nchild=( 11 12 13 14 15 16 17)
maxdepth=( 10 20 )
for i in ${maxdepth[@]}
do
  for j in ${nparent[@]}
  do
    for k in ${nchild[@]}
    do
      python BDT_FiveRating_One.py 3 test_3_${i}_${j}_${k}.txt n $i $j $k &
    done
  done
done
