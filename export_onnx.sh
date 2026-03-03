# optimum-cli export onnx \
#     --model MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor \
#     --task text-generation \
#     --trust-remote-code \
#     --device cuda --fp16 \
#     MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor_onnx

python register_minicpm_onnx.py --fp16 --device cpu --task text-generation \
    --output MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor-text-generation_int4.onnx