#!/bin/bash
for fold in 0 1 2 3 4;
do
    for nu in 0.001 0.010 0.100 1.000;
    do
        echo "$fold, $nu"
        sed -e "s#FOLD#${fold}#g" \
            -e "s#NU#${nu}#g" \
            -e "s#^work_dir.*#work_dir = '$PWD'#g" \
            run_mfm_cv.py > "run_mfm_cv_nu${nu}_f${fold}.py"
    done
done
