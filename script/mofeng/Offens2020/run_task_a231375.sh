export TASK_DATA_PATH=mofeng_data/Offens2020/task_a/231375
export MODEL_PATH=ernie/model/ERNIE_Base_en_stable-2.0.0

python -u ./ernie/run_classifier.py                                          \
           --use_cuda true                                                   \
           --for_cn  False                                                   \
           --use_fast_executor ${e_executor:-"true"}                         \
           --tokenizer ${TOKENIZER:-"FullTokenizer"}                         \
           --use_fp16 ${USE_FP16:-"false"}                                   \
           --do_train true                                                   \
           --do_val true                                                     \
           --do_test False                                                   \
           --batch_size 1                                                   \
           --init_pretraining_params ${MODEL_PATH}/params                    \
           --verbose true                                                    \
           --train_set ${TASK_DATA_PATH}/train.tsv                           \
           --dev_set   ${TASK_DATA_PATH}/dev.tsv                             \
           --vocab_path ${MODEL_PATH}/vocab.txt                              \
           --output ${TASK_DATA_PATH}/output/result.txt                      \
           --checkpoints ./checkpoints/offens2020/task_a231375_1                                  \
           --save_steps 1000                                                 \
           --weight_decay  0.0                                               \
           --warmup_proportion 0.1                                           \
           --validation_steps 1000000                                        \
           --epoch 1                                                         \
           --max_seq_len 128                                                 \
           --ernie_config_path ${MODEL_PATH}/ernie_config.json                \
           --learning_rate 1e-5                                              \
           --skip_steps 10                                                   \
           --num_iteration_per_drop_scope 1                                  \
           --num_labels 2                                                    \
           --metric 'acc_and_f1'                                             \
           --for_cn  False                                                   \
           --random_seed 1 2>&1\
