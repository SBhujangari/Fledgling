# Deployment Guide: Running on Another Computer

This guide shows how to clone this project and run the fine-tuned SLM on any computer with GPU access.

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 12GB+ VRAM (e.g., RTX 3090, RTX 4090, A100)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB free space

### Software Requirements
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher
- **Node.js**: 18.x or higher
- **Git**: Any recent version

---

## Quick Start (5 Minutes)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI_ATL25.git
cd AI_ATL25
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers peft bitsandbytes accelerate
```

### 3. Run the Example Agent

**The model will automatically download from HuggingFace:**

```bash
CUDA_VISIBLE_DEVICES=0 python example_slm_agent.py
```

**That's it!** The script will:
1. Download the fine-tuned adapter from HuggingFace (~335MB)
2. Download the base model (~4.5GB, cached after first run)
3. Run 5 example API generation tasks
4. Save traces and metrics

### 4. Start the Backend

```bash
cd backend
npm install
npm run dev
```

### 5. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

### 6. View Dashboard

Navigate to: **http://localhost:5173/slm-dashboard**

---

## How It Works

### Automatic HuggingFace Loading

The code is configured to load the model from HuggingFace by default:

```python
# In example_slm_agent.py
BASE_MODEL = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
ADAPTER = "kineticdrive/llama-structured-api-adapter"  # From HuggingFace
```

**First run:**
- Downloads adapter from HuggingFace (~335MB)
- Caches in `~/.cache/huggingface/hub/`
- Subsequent runs use cached version

**No need to commit large model files to git!**

---

## Using Local Model (Optional)

If you have the model trained locally and want to use it:

```bash
USE_LOCAL_MODEL=true python example_slm_agent.py
```

This will use `slm_swap/04_ft/adapter_llama_structured/` instead of downloading from HuggingFace.

---

## Configuration Options

### Environment Variables

#### `USE_LOCAL_MODEL`
- **Default**: `false` (uses HuggingFace)
- **Set to**: `true` to use local adapter
- **Example**: `USE_LOCAL_MODEL=true python example_slm_agent.py`

#### `CUDA_VISIBLE_DEVICES`
- **Default**: Uses all GPUs
- **Set to**: GPU ID to use specific GPU
- **Example**: `CUDA_VISIBLE_DEVICES=0` (use first GPU only)

---

## Directory Structure

```
AI_ATL25/
├── example_slm_agent.py          # Main agent script (uses HF by default)
├── eval_structured_detailed.py   # Evaluation script (uses HF by default)
├── backend/                       # Express API server
│   ├── src/
│   │   ├── routes/metrics.ts     # Metrics endpoints
│   │   └── data/slm_traces/      # Generated trace data
│   └── package.json
├── frontend/                      # React dashboard
│   ├── src/
│   │   └── pages/SLMDashboardPage.tsx
│   └── package.json
├── slm_swap/
│   ├── 02_dataset/               # Training/test data (gitignored)
│   └── 04_ft/                    # Local adapter (gitignored)
└── .gitignore                    # Excludes large files
```

### What's Committed to Git

✅ **Included:**
- Source code (Python, TypeScript, React)
- Configuration files
- Documentation
- Small datasets

❌ **Excluded (via .gitignore):**
- `slm_swap/04_ft/` - Large model files
- `*.safetensors` - Model weights
- `node_modules/` - Dependencies
- `.venv/` - Python virtual environment

---

## First-Time Setup on New Computer

### Step-by-Step

1. **Clone repo**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI_ATL25.git
   cd AI_ATL25
   ```

2. **Install Python deps**:
   ```bash
   pip install torch transformers peft bitsandbytes accelerate
   ```

3. **Run agent** (downloads model automatically):
   ```bash
   python example_slm_agent.py
   ```

4. **Install backend deps**:
   ```bash
   cd backend && npm install && cd ..
   ```

5. **Install frontend deps**:
   ```bash
   cd frontend && npm install && cd ..
   ```

6. **Start services**:
   ```bash
   # Terminal 1
   cd backend && npm run dev

   # Terminal 2
   cd frontend && npm run dev
   ```

7. **Open dashboard**:
   http://localhost:5173/slm-dashboard

---

## Model Download Details

### What Gets Downloaded

When you first run `example_slm_agent.py`:

1. **Base Model** (~4.5GB):
   - `unsloth/llama-3.1-8b-instruct-bnb-4bit`
   - Downloaded from HuggingFace
   - Cached in `~/.cache/huggingface/hub/`

2. **Fine-tuned Adapter** (~335MB):
   - `kineticdrive/llama-structured-api-adapter`
   - Downloaded from HuggingFace
   - Cached in `~/.cache/huggingface/hub/`

### Download Time

- **First run**: 5-15 minutes (depending on internet speed)
- **Subsequent runs**: Instant (uses cached models)

### Cache Location

```
~/.cache/huggingface/hub/
├── models--unsloth--llama-3.1-8b-instruct-bnb-4bit/
└── models--kineticdrive--llama-structured-api-adapter/
```

To clear cache: `rm -rf ~/.cache/huggingface/hub/`

---

## Verifying Setup

### Test Model Loading

```bash
python -c "from transformers import AutoModelForCausalLM; print('Transformers OK')"
python -c "from peft import PeftModel; print('PEFT OK')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
Transformers OK
PEFT OK
CUDA available: True
```

### Test Agent

```bash
python example_slm_agent.py
```

Expected output:
```
📦 Loading adapter from: HuggingFace (kineticdrive/llama-structured-api-adapter)
🤖 Initializing Fine-tuned API Generator...
📦 Loading model...
✅ Model loaded in X.XXs
🚀 Running 5 example API generation tasks...
...
✅ Example agent run complete!
```

### Test API

```bash
# Start backend
cd backend && npm run dev

# In another terminal
curl http://localhost:3000/api/metrics/dashboard
```

Should return JSON with dashboard data.

---

## Troubleshooting

### "CUDA out of memory"

**Problem**: GPU doesn't have enough memory

**Solutions**:
1. Close other GPU applications
2. Use smaller batch size
3. Try with single GPU: `CUDA_VISIBLE_DEVICES=0`

### "Model not found on HuggingFace"

**Problem**: Can't download from HuggingFace

**Solutions**:
1. Check internet connection
2. Verify model exists: https://huggingface.co/kineticdrive/llama-structured-api-adapter
3. Use local model: `USE_LOCAL_MODEL=true python example_slm_agent.py`

### "ModuleNotFoundError: No module named 'transformers'"

**Problem**: Python dependencies not installed

**Solution**:
```bash
pip install torch transformers peft bitsandbytes accelerate
```

### "Port 3000 already in use"

**Problem**: Backend port conflict

**Solutions**:
1. Kill process: `lsof -ti:3000 | xargs kill -9`
2. Or change port in `backend/src/server.ts`

### "No metrics available" in dashboard

**Problem**: Agent hasn't run yet

**Solution**:
```bash
python example_slm_agent.py
```

---

## Production Deployment

### Option 1: Cloud GPU (Recommended)

**Providers**:
- **Vast.ai**: ~$0.20/hour for RTX 3090
- **RunPod**: ~$0.34/hour for RTX 3090
- **AWS EC2**: g4dn.xlarge with T4 GPU

**Steps**:
1. Rent GPU instance
2. Clone repo
3. Install dependencies
4. Run agent
5. Expose ports 3000 (backend) and 5173 (frontend)

### Option 2: On-Premise Server

**Setup**:
1. Install NVIDIA drivers
2. Install CUDA toolkit
3. Clone repo and install dependencies
4. Use systemd or PM2 to keep services running

**Example systemd service** (`/etc/systemd/system/slm-backend.service`):
```ini
[Unit]
Description=SLM Backend API
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/AI_ATL25/backend
ExecStart=/usr/bin/npm run dev
Restart=always

[Install]
WantedBy=multi-user.target
```

### Option 3: Docker (Advanced)

See `docker/` directory for Dockerfile and docker-compose.yml (if available).

---

## Updating the Model

### If You Retrain the Model

1. **Upload new adapter to HuggingFace**:
   ```bash
   python slm_swap/hf_upload.py \
     slm_swap/04_ft/adapter_llama_structured \
     --repo-id kineticdrive/llama-structured-api-adapter \
     --commit-message "Updated model with better accuracy"
   ```

2. **On other computers**:
   ```bash
   # Clear cache to force re-download
   rm -rf ~/.cache/huggingface/hub/models--kineticdrive--llama-structured-api-adapter

   # Run agent (will download new version)
   python example_slm_agent.py
   ```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test SLM Agent

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install torch transformers peft bitsandbytes accelerate

    - name: Test model loading (CPU only)
      run: |
        python -c "from transformers import AutoTokenizer; print('OK')"

    # Note: Actual model loading requires GPU
```

---

## Cost Estimation

### Per Deployment

**One-time setup**:
- Download base model: Free (HuggingFace)
- Download adapter: Free (HuggingFace)
- Time: 5-15 minutes

**Running costs**:
- **Cloud GPU**: $0.20-0.50/hour
- **On-premise**: Electricity + hardware depreciation (~$83/month)
- **API calls**: $0 (no per-token costs)

### vs Azure GPT

**100K requests/month**:
- **Azure GPT-4**: ~$15,000/month
- **Our SLM (cloud)**: ~$150/month (continuous)
- **Our SLM (on-premise)**: ~$83/month

**180x cost reduction at scale!**

---

## Support & Documentation

- **Setup Issues**: See troubleshooting section above
- **Model Details**: `INVESTOR_PITCH.md`
- **Demo Guide**: `SLM_DEMO_README.md`
- **Technical Details**: `FINE_TUNING_EVALUATION_FINDINGS.md`

---

## Summary: Running on New Computer

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/AI_ATL25.git
cd AI_ATL25

# 2. Install Python deps
pip install torch transformers peft bitsandbytes accelerate

# 3. Run agent (auto-downloads from HuggingFace)
python example_slm_agent.py

# 4. Install & start backend
cd backend && npm install && npm run dev &

# 5. Install & start frontend
cd frontend && npm install && npm run dev &

# 6. Open http://localhost:5173/slm-dashboard
```

**Total time: ~10 minutes (excluding model download)**

---

## Key Points

✅ **No large files in git** - Model downloads from HuggingFace automatically
✅ **Zero configuration** - Works out of the box on any computer with GPU
✅ **Fast iteration** - Update model on HuggingFace, clear cache, re-run
✅ **Production ready** - Same code works locally and in cloud
✅ **Cost efficient** - Free model hosting on HuggingFace

**You can now push to git and deploy anywhere!** 🚀
