export HF_HOME='/raid0-data/yifan'
export CUDA_VISIBLE_DEVICES=0
dataset=wikitext2

# batch_size=256, 512, 1024, 2048

# batch_size=128

batch_size=128
seqlen=256

python run_benchmark.py \
        --model mistralai/Mistral-7B-v0.1 \
        --device cuda:0 \
        --seqlen $seqlen \
        --batch-size $batch_size \
        --eval-dataset $dataset 

# for sparsity in 88 77 66 55 44
# do
# python run_benchmark.py \
#         --model mistralai/Mistral-7B-v0.1 \
#         --sliced-model-path ./out_models/mistral-uniform/$dataset/mistral-7b-$sparsity \
#         --seqlen $seqlen \
#         --device cuda:0 \
#         --batch-size $batch_size \
#         --eval-dataset $dataset
# done

# for sparsity in 90 80 70 60 50
# do
# python run_benchmark.py \
#         --model mistralai/Mistral-7B-v0.1 \
#         --sliced-model-path ./out_svdllm/checkpoints/svd_llm_mistral_7b_$sparsity/model.pt \
#         --device cuda:0 \
#         --batch-size $batch_size \
#         --eval-dataset $dataset
# done

# for sparsity in 0.2 0.3 0.4 0.45 0.55
# do
# python run_benchmark.py \
#         --model mistralai/Mistral-7B-v0.1 \
#         --sliced-model-path ./out_slicegpt/mistral-7b/$dataset/Llama-2-7b-hf_$sparsity \
#         --device cuda:0 \
#         --batch-size $batch_size \
#         --eval-dataset $dataset
# done

