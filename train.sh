#!
MODEL_PATH=$1
OUTPUT_PATH=$2
TRAIN_FILE=$3
TEST_FILE=$4

BLOCK_SIZE=360
LEARNING_RATE=5e-8
SAVE_STEPS=1000

LANGUAGE_MODELING=../transformers/examples/run_language_modeling.py

python $LANGUAGE_MODELING \
    --output_dir=$OUTPUT_PATH \
    --model_type=gpt2 \
    --model_name_or_path=$MODEL_PATH \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --block_size $BLOCK_SIZE \
    --learning_rate $LEARNING_RATE \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --save_steps $SAVE_STEPS \
    --no_cuda
# remove --no_cuda if you have a beefy GPU to train the model on
