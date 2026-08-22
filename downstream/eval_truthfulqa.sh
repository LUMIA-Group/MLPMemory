#!/bin/bash

MODEL=/path/to/models/Mistral-7B-v0.3
MODEL_NAME=mistral
# MLP Memory can be downloaded from https://huggingface.co/Rubin-Wei/MLPMemory-Mistral-wikipedia
KNN_GENERATOR_PATH=/path/to/mlp/memory

TASKS=("truthfulqa")
DATA_ROOT=downstream/data

# Fixed lambda for TruthfulQA MLPMemory evaluation.
LMBDA="0.75"
GPU_ID=0

echo "======================================="
echo "Starting TruthfulQA evaluation"
echo "Model: $MODEL_NAME"
echo "MLPMemory lambda: $LMBDA"
echo "======================================="

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "======================================="
    echo "Evaluating task: $TASK"
    echo "======================================="

    if [[ $TASK == "truthfulqa" ]]; then
        DATASET_NAME=${DATA_ROOT}/TruthfulQA/TruthfulQA.csv
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
            python -m downstream.eval_truthfulqa \
              --model-name-or-path ${MODEL} \
              --data-path ${DATASET_NAME} \
              --output-path ${OUTPUT_DIR}/output.json \
              --mode base \
              --dtype float16 \
              2>&1 | tee ${LOG_FILE}
        elif [[ $MODE == "mlpmemory" ]]; then
            OUTPUT_DIR=downstream/results/${TASK}/${MODEL_NAME}/mlpmemory/lambda_${LMBDA}
            LOG_FILE=downstream/logs/${TASK}_${MODEL_NAME}_mlpmemory_lambda_${LMBDA}_gpu_${GPU_ID}.log

            echo "[GPU $GPU_ID] Starting MLPMemory evaluation for $TASK with lambda=$LMBDA"
            python -m downstream.eval_truthfulqa \
              --model-name-or-path ${MODEL} \
              --data-path ${DATASET_NAME} \
              --output-path ${OUTPUT_DIR}/output.json \
              --mode mlpmemory \
              --knn-generator-path ${KNN_GENERATOR_PATH} \
              --lmbda ${LMBDA} \
              --knn-temp 1.0 \
              --dtype float16 \
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
echo "All TruthfulQA evaluations completed!"
echo "======================================="
