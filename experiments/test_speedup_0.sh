export HF_HOME='/raid0-data/yifan'
export CUDA_VISIBLE_DEVICES=1
dataset=wikitext2


# for sparsity in 0.2 0.3 0.4 0.45
# do
# python run_benchmark.py \
#         --model meta-llama/Llama-2-7b-hf \
#         --device cuda:0 \
#         --ntokens 128 \
#         --batch-size 256 \
#         --sliced-model-path /raid0-data/yifan/out_slicegpt/llama-2-7b-uniform/$dataset/Llama-2-7b-hf_$sparsity \
#         --eval-dataset $dataset \
#         --no-wandb 
# done

for sparsity in 88 76 128 52 40
do
python run_benchmark.py \
        --model meta-llama/Llama-2-7b-hf \
        --sliced-model-path /raid0-data/yifan/out_models/llama-2-7b/no_QK/Llama-2-7b-hf_$sparsity \
        --device cuda:0 \
        --ntokens 128 \
        --batch-size 256 \
        --eval-dataset $dataset
done

for sparsity in 90 80 70 60 50
do
python run_benchmark.py \
        --model meta-llama/Llama-2-7b-hf \
        --sliced-model-path /raid0-data/yifan/out_svdllm/checkpoints/svd_llm_llama_2_7b_$sparsity/model.pt \
        --device cuda:0 \
        --ntokens 128 \
        --batch-size 256 \
        --eval-dataset $dataset
done


for sparsity in 0.2 0.3 0.4 0.45 0.55
do
python run_benchmark.py \
        --model meta-llama/Llama-2-7b-hf \
        --sliced-model-path /raid0-data/yifan/out_slicegpt/llama-2-7b-uniform/$dataset/Llama-2-7b-hf_$sparsity \
        --device cuda:0 \
        --ntokens 128 \
        --batch-size 256 \
        --eval-dataset $dataset
done

# --sliced-model-path /raid0-data/yifan/out_svdllm/checkpoints/svd_llm_llama_2_7b_$sparsity/model.pt \
# --sliced_model_path /raid0-data/yifan/out_slicegpt/llama-2-7b-uniform/$dataset/Llama-2-7b-hf_$sparsity \
# --sliced-model-path /raid0-data/yifan/out_models/llama-2-7b-uniform/no_QK/Llama-2-7b-hf_$sparsity \
