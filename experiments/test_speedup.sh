export HF_HOME='/raid0-data/yifan'
export CUDA_VISIBLE_DEVICES=7
dataset=wikitext2


batch_size=128

# for ratio in 88 76 64 52 40
# do
# python run_benchmark.py \
#         --model meta-llama/Llama-2-7b-hf \
#         --sliced-model-path ./out_models/llama-2-7b/Llama-2-7b-hf_${ratio}/pytorch_model.bin \
#         --device cuda:0 \
#         --batch-size $batch_size \
#         --eval-dataset $dataset 2>&1 | tee -a log_2.txt
# done

python run_benchmark.py \
        --model meta-llama/Llama-2-7b-hf \
        --device cuda:0 \
        --batch-size $batch_size \
        --eval-dataset $dataset 2>&1 | tee -a log_2.txt

# for ratio in 0.2 0.3 0.4 0.45 0.55
# do
# python run_benchmark.py \
#         --model meta-llama/Llama-2-7b-hf \
#         --sliced-model-path ./out_slicegpt/llama-2-7b/$dataset/Llama-2-7b-hf_$ratio \
#         --device cuda:0 \
#         --batch-size $batch_size \
#         --eval-dataset $dataset 2>&1 | tee -a log_2.txt
# done

# for ratio in 90 80 70 60 50
# do
# python run_benchmark.py \
#         --model meta-llama/Llama-2-7b-hf \
#         --sliced-model-path ./out_svdllm/checkpoints/svd_llm_llama_2_7b_$ratio/model.pt \
#         --device cuda:0 \
#         --batch-size $batch_size \
#         --eval-dataset $dataset 2>&1 | tee -a log_2.txt
# done
