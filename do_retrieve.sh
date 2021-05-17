python3 ./data_preprocess/get_ob_context.py --file $1
python3 ./data_preprocess/get_comm_context_AMR.py --file $1
python3 ./data_preprocess/sort_para_with_AMR.py  --file $1
rm -rf $1\0
rm -rf $1\1
rm -rf $1\2