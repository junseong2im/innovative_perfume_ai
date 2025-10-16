"""
Test Qwen 2.5-7B Model with GPU
실제 딥러닝 모델 다운로드 및 GPU 실행 테스트
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

print("="*60)
print("QWEN 2.5-7B GPU TRAINING TEST")
print("="*60)
print()

# Check GPU
print("[1] Checking GPU availability...")
print(f"    PyTorch version: {torch.__version__}")
print(f"    CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    CUDA version: {torch.version.cuda}")
print()

# Load model
print("[2] Loading Qwen/Qwen2.5-7B-Instruct model...")
print("    This will download ~15GB on first run...")
print("    Please wait 5-10 minutes for download...")
print()

start_time = time.time()

try:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16,  # Use FP16 for faster inference
        device_map="auto",  # Automatically use GPU
        trust_remote_code=True
    )

    load_time = time.time() - start_time
    print(f"[OK] Model loaded in {load_time:.1f}s")
    print(f"    Model device: {model.device}")
    print(f"    Model dtype: {model.dtype}")
    print()

    # Test inference
    print("[3] Testing inference with sample prompt...")
    print()

    prompt = """분석해줘: "여름에 어울리는 상쾌한 시트러스 향수를 만들고 싶어요"

다음 형식으로 추출:
{
  "style": "fresh",
  "season": "summer",
  "notes": ["citrus", "fresh"],
  "intensity": 0.7,
  "masculinity": 0.5
}"""

    inference_start = time.time()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

    inference_time = time.time() - inference_start

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"[OK] Inference complete in {inference_time:.1f}s")
    print()
    print("Response:")
    print("-"*60)
    print(response[len(prompt):])  # Print only the generated part
    print("-"*60)
    print()

    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory:")
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Reserved:  {memory_reserved:.2f} GB")
        print()

    print("="*60)
    print("VERDICT")
    print("="*60)
    print()
    print("[PASS] Qwen 2.5-7B GPU inference successful!")
    print(f"  - Load time: {load_time:.1f}s")
    print(f"  - Inference time: {inference_time:.1f}s")
    print(f"  - Model size: ~7 billion parameters")
    print(f"  - Running on: {model.device}")

except Exception as e:
    print(f"[ERROR] Failed to load or run model: {e}")
    import traceback
    traceback.print_exc()
