#!/bin/bash

#SemEval
python re_ddagat_main.py --do_test --task_name semeval --data_dir ./data/semeval/ --model_path MODELNAME --do_lower_case


#TACRED
python re_ddagat_main.py --do_test --task_name tacred --data_dir ./data/tacred/ --model_path MODELNAME --do_lower_case

