#!/bin/bash

#SemEval
python re_ddagat_main.py --do_train --do_eval --task_name semeval --data_dir ./data/semeval/ --model_path ./bert/bert-base-uncased/ --model_name MODELNAME --do_lower_case


#TACRED
python re_ddagat_main.py --do_train --do_eval --task_name tacred --data_dir ./data/tacred/ --model_path ./bert/bert-base-uncased/ --model_name MODELNAME --do_lower_case

