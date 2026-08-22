#!/bin/bash

MODEL=/path/to/models/Mistral-7B-v0.3
MODEL_NAME=mistral
# MLP Memory can be downloaded from https://huggingface.co/Rubin-Wei/MLPMemory-Mistral-wikipedia
KNN_GENERATOR_PATH=/path/to/mlp/memory

TASKS=("webqa" "triviaqa")
DATA_ROOT=downstream/data

WEBQA_LMBDA="0.75"
TRIVIAQA_LMBDA="0.60"
GPU_ID=0

echo "======================================="
echo "Starting WebQA and TriviaQA evaluation"
echo "Model: $MODEL_NAME"
echo "WebQA lambda: $WEBQA_LMBDA"
echo "TriviaQA lambda: $TRIVIAQA_LMBDA"
echo "======================================="

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "======================================="
    echo "Evaluating task: $TASK"
    echo "======================================="

    if [[ $TASK == "webqa" ]]; then
        DATASET_NAME=${DATA_ROOT}/webqa/test.jsonl
        LMBDA=${WEBQA_LMBDA}
    elif [[ $TASK == "triviaqa" ]]; then
        DATASET_NAME=${DATA_ROOT}/triviaqa/test.jsonl
        LMBDA=${TRIVIAQA_LMBDA}
    else
        echo "Error: Unknown task $TASK"
        exit 1
    fi

    run_evaluation() {
        local MODE=$1

        export CUDA_VISIBLE_DEVICES=$GPU_ID
        mkdir -p downstream/logs

        if [[ $MODE == "base" ]]; then
            OUTPUT_DIR=downstream/results/${TASK}/${MODEL_NAME}/base
            LOG_FILE=downstream/logs/${TASK}_${MODEL_NAME}_base_gpu_${GPU_ID}.log

            echo "[GPU $GPU_ID] Starting base evaluation for $TASK"
            python -m downstream.eval_webqa_triviaqa \
              --data ${TASK} \
              --data-path ${DATASET_NAME} \
              --model-name-or-path ${MODEL} \
              --results-dir ${OUTPUT_DIR} \
              --mode base \
              --eval-batch-size 4 \
              --max-new-tokens 100 \
              --dtype bfloat16 \
              2>&1 | tee ${LOG_FILE}
        elif [[ $MODE == "mlpmemory" ]]; then
            OUTPUT_DIR=downstream/results/${TASK}/${MODEL_NAME}/mlpmemory/lambda_${LMBDA}
            LOG_FILE=downstream/logs/${TASK}_${MODEL_NAME}_mlpmemory_lambda_${LMBDA}_gpu_${GPU_ID}.log

            echo "[GPU $GPU_ID] Starting MLPMemory evaluation for $TASK with lambda=$LMBDA"
            python -m downstream.eval_webqa_triviaqa \
              --data ${TASK} \
              --data-path ${DATASET_NAME} \
              --model-name-or-path ${MODEL} \
              --results-dir ${OUTPUT_DIR} \
              --mode mlpmemory \
              --knn-generator-path ${KNN_GENERATOR_PATH} \
              --lmbda ${LMBDA} \
              --knn-temp 1.0 \
              --eval-batch-size 4 \
              --max-new-tokens 100 \
              --dtype bfloat16 \
              2>&1 | tee ${LOG_FILE}
        else
            echo "Error: Unknown mode $MODE"
            exit 1
        fi

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "[GPU $GPU_ID] Evaluation failed for $TASK ($MODE). See ${LOG_FILE}"
            exit 1
        fi
        echo "[GPU $GPU_ID] Completed evaluation for $TASK ($MODE)"
    }

    run_evaluation base
    run_evaluation mlpmemory
done

echo "======================================="
echo "All WebQA and TriviaQA evaluations completed!"
echo "======================================="
