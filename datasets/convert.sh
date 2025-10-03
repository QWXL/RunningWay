echo starting
python ../json2binidx_tool/tools/preprocess_data.py \
    --input ./outputs/shuffled-mini-test.jsonl\
    --output-prefix ./data/mini-test \
    --vocab ../1.0/rwkv_vocab_v20230424.txt \
    --dataset-impl mmap \
    --tokenizer-type RWKVTokenizer 
 --append-eod