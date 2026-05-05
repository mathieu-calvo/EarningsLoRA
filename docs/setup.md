# Setup — accounts, keys, first training run

End-to-end walkthrough: from zero accounts to a populated `bench.json` and a
live HF Space. Everything here uses free tiers; total cost is $0.

Estimated time: 30-45 min of setup + ~2 hr of (mostly hands-off) training.

---

## 0. Prerequisites

- A GitHub account (you've cloned this repo).
- A web browser. That's it for the setup phase — the training itself runs on Kaggle, not your laptop.

---

## 1. HuggingFace account, Llama 3.2 access, and `HF_TOKEN`

Llama 3.2 is **gated** — Meta requires an approval step before you can download
it. Approval is usually instant but can take a few hours.

1. Create an account at <https://huggingface.co/join>.
2. Request access to Llama 3.2 3B Instruct:
   - Go to <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>.
   - Click **"Access repository"** at the top.
   - Fill in the Meta form (name, affiliation, intended use). Submit.
   - You'll get an email when approved. Refresh the model page — the download tab should now be visible.
3. Create a token:
   - <https://huggingface.co/settings/tokens> → **"Create new token"**.
   - Type: **Read** (sufficient for training; you'll create a separate write token in §6 for publishing).
   - Name it something like `earningslora-read`.
   - **Copy the token now** — you won't see it again.

Save the token somewhere temporary; you'll paste it into Kaggle in §4.

---

## 2. Google AI Studio and `GOOGLE_API_KEY`

Used for the frontier baseline (Gemini 2.5 Flash) and the LLM-as-judge.
Free tier: 1500 requests/day, plenty for a 50-row hold-out.

1. Go to <https://aistudio.google.com/app/apikey> (sign in with a Google account).
2. **"Create API key"** → pick "Create API key in new project" if you don't have a Google Cloud project.
3. Copy the key (`AIza...`).

The eval harness caches all Gemini calls to disk, so you only spend quota once
per unique (transcript, prompt-version) pair.

---

## 3. Weights & Biases and `WANDB_API_KEY`

Used to track the training run (loss curves, eval metrics, GPU utilisation).
Optional — you can skip and set `WANDB_MODE=offline`, but the dashboards are
worth the 60-second setup.

1. Sign up at <https://wandb.ai/signup>.
2. <https://wandb.ai/authorize> → copy the long key.
3. (Optional) Create a project named `earningslora` at <https://wandb.ai/new-project> so runs land in the right place.

---

## 4. Kaggle account, GPU enablement, and Secrets

Kaggle gives **30 GPU hours/week** for free on T4 (or P100). That's enough for
~15 training runs, plus eval.

1. Sign up at <https://www.kaggle.com/account/login>.
2. **Verify your phone number** — required to access GPU accelerators. Account → Settings → Phone Verification.
3. Add your three secrets:
   - Open any notebook (or create a new one) → **Add-ons → Secrets**.
   - Add `HF_TOKEN` (from §1), `GOOGLE_API_KEY` (from §2), `WANDB_API_KEY` (from §3).
   - Toggle each one **on** for the notebook you'll use to train.

---

## 5. Run the training notebook on Kaggle

Two paths: import the notebook from this repo (recommended), or paste the cells
into a new Kaggle notebook.

### Path A — import `notebooks/03_qlora_training.ipynb`

1. <https://www.kaggle.com/code> → **"New Notebook"** → **File → Import Notebook** → upload `notebooks/03_qlora_training.ipynb` from your local clone.
2. Right sidebar:
   - **Accelerator**: **GPU T4 x2** (or single T4 — both work).
   - **Internet**: **On** (required to clone the repo and pull the model).
   - **Persistence**: **Files only** (so checkpoints survive restarts).
3. Make sure the three secrets from §4 are toggled on (sidebar → Add-ons → Secrets).
4. Uncomment the Kaggle Secrets block in the second cell:
   ```python
   from kaggle_secrets import UserSecretsClient
   secrets = UserSecretsClient()
   os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
   os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
   os.environ["GOOGLE_API_KEY"] = secrets.get_secret("GOOGLE_API_KEY")
   ```
5. **Run All**. The notebook will:
   - Clone the repo and `pip install -e ".[train,eval]"`.
   - Pull the chat-formatted dataset from `mathieu-calvo/ectsum-chat`.
   - Sanity-check the chat template renders correctly.
   - Train for ~90-120 min (you can close the tab — Kaggle keeps it running for up to 9 hr).
   - Run a single sanity-generation against the hold-out at the end.
6. When training completes, **download `runs/latest/adapter/`** from the notebook's "Output" tab — that's your ~50 MB LoRA adapter.

### Path B — Colab fallback

Use this if Kaggle's GPU queue is full or your account is unverified.

1. <https://colab.research.google.com> → upload `notebooks/03_qlora_training.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Replace the Kaggle Secrets block with:
   ```python
   from google.colab import userdata
   os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
   os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
   os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
   ```
4. Add the three secrets via the **🔑 key icon** in the left sidebar.
5. **Runtime → Run all**.
6. Colab sessions cap at 12 hr, so you have headroom. If you hit the cap mid-run, restart with `train(..., resume_from="runs/latest/checkpoint-<N>")`.

### Pre-training checklist

Before clicking "Run All", verify in this order:
- [ ] Llama 3.2 access granted (check the model page shows the file list, not a "Request access" button).
- [ ] All three secrets visible to the notebook (the assertion in cell 2 will fail loudly if `HF_TOKEN` is missing).
- [ ] GPU accelerator selected (the notebook will silently fall back to CPU and take ~80x longer otherwise).
- [ ] Internet on (Kaggle off-by-default).

---

## 6. After training: eval + publish

These steps run on your **local machine** (no GPU needed for eval — the
predictions are JSONL files, the judge is API-based).

1. Copy the downloaded adapter into the local repo:
   ```bash
   mkdir -p runs/latest
   # Unzip the Kaggle output into runs/latest/adapter/
   ```
2. Create a local `.env`:
   ```bash
   cp .env.example .env
   # Fill in HF_TOKEN, GOOGLE_API_KEY (WANDB_API_KEY optional locally).
   ```
3. Build the chat dataset + frozen hold-out (one-time):
   ```bash
   python scripts/prepare_dataset.py
   ```
4. Run the bench:
   ```bash
   python scripts/evaluate.py --adapter-dir runs/latest/adapter
   ```
   This writes `reports/bench.json` and updates the README headline table.
5. Create a **write-scoped** HF token at <https://huggingface.co/settings/tokens> ("Create new token" → **Write**), and replace `HF_TOKEN` in `.env` with it.
6. Publish to the Hub:
   ```bash
   python scripts/publish.py --all --adapter-dir runs/latest/adapter --dry-run   # preview
   python scripts/publish.py --all --adapter-dir runs/latest/adapter             # ship
   ```
   This pushes the adapter (with auto-rendered model card), the dataset, and the Space.

The merged-bf16 weights for the Space need to be produced inside a GPU
session (the merge loads the base model in bf16 ≈ 6 GB). Easiest path: do it
in the same Kaggle session, after the training cell:

```python
from earningslora.inference.merge import merge_and_save
merge_and_save(settings.base_model, "runs/latest/adapter", "runs/latest/merged")
# Then upload runs/latest/merged/ to the Hub model repo (or replace the adapter
# repo contents with the merged weights so the Space loads them directly).
```

---

## 7. Configure the HF Space (one-time)

For the Gradio demo to render the Gemini column, the Space needs its own
secrets:

1. Once `scripts/publish.py --space` has created the Space, open it on the Hub.
2. **Settings → Repository secrets** (under the title) → add:
   - `HF_TOKEN` — your read token (Llama is gated; the Space cold-start needs to download it).
   - `GOOGLE_API_KEY` — same key from §2.
3. **Settings → Hardware** → confirm **ZeroGPU** is selected.
4. Trigger a rebuild (push a commit or click **Restart Space**).

Cold-start ~15-30s. Subsequent requests warm.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `403` pulling Llama on Kaggle | Gated-access not yet approved, or `HF_TOKEN` typo | Wait for the approval email; re-paste the token into Kaggle Secrets. |
| `bitsandbytes` import error on Windows | bitsandbytes is GPU-only and skipped on Windows | Train on Kaggle/Colab; local Windows runs the eval/publish path only. |
| OOM during training | Kaggle assigned a smaller GPU than expected | Lower `--max-seq-len` to 2048 or `--grad-accum` to 4 in `scripts/train.py`. |
| Frontier column blank in the Space | `GOOGLE_API_KEY` not set as a Space secret | See §7. |
| README table didn't update after eval | bench.json wrote successfully but markers missing in README | The markers `<!-- BENCH:HEADLINE:BEGIN -->` / `<!-- BENCH:HEADLINE:END -->` must surround the table in `README.md`. |
| `HF_TOKEN does not have permission to write` | You're using the read-scoped token from §1 | Create a separate Write token (§6 step 5). |

---

## Cost summary (for the receipts)

| Service | Free-tier limit | Used here |
|---|---|---|
| Kaggle T4 | 30 hr/week | ~2 hr per training run |
| HF Hub | Unlimited public repos | 3 (adapter, dataset, space) |
| HF ZeroGPU | Public spaces, ephemeral H200 | Per-request, idle = 0 |
| Google AI Studio | 1500 req/day | ~150 (50 hold-out × 3 calls: frontier + 2 judge pairs) |
| W&B personal | Unlimited public projects | 1 project |

Total monthly cost: **$0**.
