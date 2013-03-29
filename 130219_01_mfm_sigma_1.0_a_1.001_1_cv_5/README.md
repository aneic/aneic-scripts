Steps to run
--

1. Get aneic-core 

        git clone https://github.com/aneic/aneic-core

2. Generate job scripts

        . ./gen_jobs_cv.sh

3. Run each job script (preferably on a cluster)

        for job in run_mfm_cv_nu*.sh
        do
            python $job
        done
    
4. Generate html output

        python ../write_html.py -o html/ results/*best*

5. Run post-processing interactive analysis

        ipython -i ./analysis_mfm_cv.py
