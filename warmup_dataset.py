
import os
import sys
from audiobench.dataset import Dataset

# 添加项目路径
sys.path.append("/home/jinchao/runtao/LLM-Pruner")

def warmup():
    DATASET_ID_CALIB = "imda_part2_asr_test"
    print(f"Starting dataset warmup for {DATASET_ID_CALIB}...")
    
    # 这里触发加载和预处理，由于 Hugging Face datasets 的机制，
    # 第一次运行会生成缓存文件。
    try:
        # nsamples 设为 1 即可，重点是触发类内部的初始化加载逻辑
        ds = Dataset(DATASET_ID_CALIB, 1)
        print("Warmup successful! Cache is now ready.")
    except Exception as e:
        print(f"Warmup encountered an issue (possibly already cached): {e}")

if __name__ == "__main__":
    warmup()
