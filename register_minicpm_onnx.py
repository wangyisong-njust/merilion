from LLMPruner.models.hf_MiniCPM.onnx_config_minicpm import MiniCPMOnnxConfig
from pathlib import Path
from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from optimum.exporters.onnx import validate_model_outputs
from transformers import AutoModelForCausalLM
import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="./MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor")
parser.add_argument("--output", type=str, default="./MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor/model.onnx")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--quantized_model", type=str, default="./MiniCPM-checkpoints/MiniCPM-2B-128k-pruned-bl-0.3-taylor_int4")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--task", type=str, default="text-generation")
args = parser.parse_args()


# Register MiniCPM model type
TasksManager._SUPPORTED_MODEL_TYPE["minicpm"] = {
    "onnx": {
        "text-generation": MiniCPMOnnxConfig,
        "text-generation-with-past": MiniCPMOnnxConfig,
    }
}

if args.quantized_model:
    from accelerate import init_empty_weights
    from optimum.gptq import load_quantized_model
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, trust_remote_code=True)
    empty_model.tie_weights()
    model = load_quantized_model(empty_model, save_folder=args.quantized_model, device_map="auto", disable_exllama=True)
else:
    # Update main export code
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

onnx_config_constructor = TasksManager.get_exporter_config_constructor(
    exporter="onnx",
    model=model,
    task=args.task,
    library_name="transformers",
)
onnx_config = onnx_config_constructor(model.config)

onnx_inputs, onnx_outputs = export(
    model=model,
    output=Path(args.output),
    device=args.device,
    dtype="fp16" if args.fp16 else "fp32",
    config=onnx_config
)



validate_model_outputs(
    onnx_config, model, args.output, onnx_outputs, onnx_config.ATOL_FOR_VALIDATION
)