# Fine-Tuning Open-Source Models for **Function Calling** (Unsloth + Docker)

> A compact, step-by-step cheat sheet distilled from the Towards AI guide, plus the essential Unsloth docs links.

---

## TL;DR

- Use **Unsloth’s Docker image** to avoid “dependency hell.”
- Train an **Llama-family** (or any HF Transformers) model with **LoRA/QLoRA** on a function-calling dataset (e.g., **Hermes Function Calling v1**).
- Validate via JSON-schema-constrained generation and a small eval set.
- Export LoRA, merge if needed, and push to HF Hub.

---

## 1) Prereqs

- **NVIDIA GPU** + recent driver and Docker with GPU access (`nvidia-container-toolkit`).
- **Docker** ≥ 24.x.
- (Optional) **Hugging Face** token for dataset/model pulls & pushing.

---

## 2) Pull & Run the Unsloth Docker

```bash
# Pull official image
docker pull unsloth/unsloth:latest

# Start a GPU-enabled container with your HF token mounted as env
docker run --gpus all -it --shm-size=16g   -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN   -v $PWD:/workspace   --workdir /workspace   unsloth/unsloth:latest bash
```

---

## 3) Pick a Base Model

Common choices (instruct variants recommended):

```text
meta-llama/Llama-3.1-8B-Instruct
mistralai/Mistral-7B-Instruct-v0.3
Qwen/Qwen2.5-7B-Instruct
```

Unsloth supports “any model that works in Transformers,” including 4-bit quantized training.

---

## 4) Dataset Shape (Function Calling)

Train the model to **emit a structured tool call** (name + JSON args). Example:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant that uses tools."},
    {"role": "user", "content": "Order 3 apples, 2 breads, 1 gallon milk at Safeway SF."}
  ],
  "tools": [
    {
      "name": "place_safeway_order",
      "description": "Place a grocery order at a Safeway store.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string"},
          "items": {"type": "array", "items": {"type": "string"}},
          "quantity": {"type": "array", "items": {"type": "integer"}}
        },
        "required": ["location", "items", "quantity"]
      }
    }
  ],
  "answer": {
    "tool_call": {
      "name": "place_safeway_order",
      "arguments": {
        "location": "San Francisco, CA",
        "items": ["apples","bread","milk"],
        "quantity": [3,2,1]
      }
    }
  }
}
```

---

## 5) Quick Training Script (LoRA/QLoRA)

Inside the container:

```bash
pip install -U "unsloth[prod]" datasets accelerate trl jsonschema
```

```python
# train_function_call.py
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
import json

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "path/to/func_call_dataset.jsonl"

def format_example(ex):
    sys = ex["messages"][0]["content"]
    user = ex["messages"][1]["content"]
    target = json.dumps(ex["answer"]["tool_call"], ensure_ascii=False)
    prompt = f"<|system|>\n{sys}\n<|user|>\n{user}\n<|assistant|>\n"
    return {"prompt": prompt, "label": target}

ds = load_dataset("json", data_files=DATA_PATH, split="train").map(format_example)

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,
    use_gradient_checkpointing=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear"
)

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=25,
    save_strategy="epoch",
    bf16=True
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    dataset_text_field="prompt",
    max_seq_length=2048,
    packing=False,
    args=args,
    response_field="label"
)

trainer.train()
model.save_pretrained("lora_out")
tokenizer.save_pretrained("lora_out")
```

---

## 6) Constrained Inference (JSON Schema)

```python
from jsonschema import validate
import json, re

def extract_json(text):
    m = re.search(r"\{.*\}\s*$", text, re.S)
    return json.loads(m.group(0)) if m else None
```

---

## 7) Evaluate Quickly

- **Exact tool name** match.
- **Arguments JSON**: valid, schema-correct, semantically right.
- Track **Strict JSON Rate**, **Schema-valid Rate**, **Task Pass Rate**.

---

## 8) Merge & Export (Optional)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="bfloat16")
merged = PeftModel.from_pretrained(base, "lora_out").merge_and_unload()
merged.save_pretrained("merged_model")
```

Push to HF:

```bash
huggingface-cli login
python - << 'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("merged_model")
t = AutoTokenizer.from_pretrained("merged_model")
m.push_to_hub("your-hf-username/llama-funcall-8b-merged")
t.push_to_hub("your-hf-username/llama-funcall-8b-merged")
PY
```

---

## 9) Upload LoRA / Eval Artifacts to Hugging Face

`git push` will fail for files >100 MB (e.g., `adapter_model.safetensors`). Ship heavy artifacts to the Hub instead:

1. Authenticate once via `huggingface-cli login` **or** set `HUGGING_FACE_HUB_TOKEN`.
2. Run the uploader with your desired repo id (set `--private` to keep it hidden until you are ready to flip public).

```bash
python slm_swap/hf_upload.py \
  slm_swap/04_ft/adapter_structured \
  slm_swap/04_ft/adapter_toolcall \
  --repo-id your-username/slm-adapters \
  --private \
  --auto-subdir \
  --commit-message "sync adapters $(date +%Y-%m-%d)"
```

- Use `--repo-type dataset` to push evaluation outputs or JSONL corpora, or stick with the default `model` type for adapters.
- Pass `--path-in-repo adapters/toolcall` (for example) to control the destination folder; otherwise `--auto-subdir` keeps each local folder under its own name.
- The script is idempotent; rerunning it overwrites/updates only the uploaded paths, so it works as a drop-in replacement for storing large binaries in Git.

---

## 9) Common Pitfalls & Fixes

- **Malformed JSON** → “return only valid JSON” + schema validation.
- **Overfitting** → Keep dev split; train only on tool_call.
- **OOM** → Use QLoRA + checkpointing + lower `r`.

---

## 10) Minimal Inference Snippet

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json

model_id = "your-hf-username/llama-funcall-8b-merged"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval()

prompt = "<|system|>\nYou are a helpful assistant that uses tools.\n<|user|>\nOrder 3 apples for SF.\n<|assistant|>\n"
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=128)
text = tok.decode(out[0], skip_special_tokens=True)
print(text)
```

---

## Notes for Codex Usage

- Save as `FUNCTION_CALLING_FINETUNE.md`.
- Create a `Makefile` or bash script to:
  1. Pull Docker
  2. Run container
  3. Execute training script
  4. Run eval + push to HF

---
