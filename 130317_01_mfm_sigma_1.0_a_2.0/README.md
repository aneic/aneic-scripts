Steps to run
--

1. Get aneic-core 

        git clone https://github.com/aneic/aneic-core

2. Run mfm model 

        python run_mfm.py

3. Generate html output

        python ../write_html.py -o html/ results/*best*

4. Run post-processing interactive analysis

        ipython -i ./analysis_mfm.py
