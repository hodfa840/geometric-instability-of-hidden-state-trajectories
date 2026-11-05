# VANE: Geometric Instability of Hidden-State Trajectories Predicts Reasoning Failures in LLMs

Code for the NeurIPS 2026 paper *Geometric Instability of Hidden-State Trajectories Predicts
Reasoning Failures in Large Language Models* (Fakharzadehjahromy, Bueff, Heintz, Tiger, Li, Li, Geng).
Paper page: https://neurips.cc/virtual/2026/loc/sydney/poster/154169
If you use this code, please cite the paper ([BibTeX below](#citation)).

VANE reads the trajectory of every generated token's hidden state across the transformer layers
and summarises its geometry with five parameter-free finite-difference descriptors
(**velocity, jerk, curvature, geodesic deviation, token coherence**). A Gradient Boosting
classifier on these descriptors predicts whether the model's answer is correct, from the same
single forward pass that produced it.

```
vane/
  metrics.py      the five metrics and their token aggregations (Section 3.2)
  features.py     feature vector (Section 3.3, Algorithm 1)
  classifier.py   Gradient Boosting, Optuna, nested cross-validation
  extraction.py   greedy generation and the single scoring forward pass
  evaluation.py   log-probability baseline, selective prediction, run I/O
  tasks/          GSM8K, MATH-500, HumanEval, CRUXEval, BBH: prompts and answer checking
scripts/
  extract.py      (GPU)  generate and extract VANE profiles for one model x benchmark
  evaluate.py     (CPU)  VANE AUROC, ablation, log-probability, selective prediction
configs/models.json      the six models of the paper
tests/                   unit tests
```

## Installation

```bash
conda create -n vane python=3.11 -y && conda activate vane
pip install -r requirements.txt
pip install -e .
pytest -q          # unit tests, CPU only
```

Llama-3 and Gemma-3 are gated on Hugging Face: accept their licences and run `huggingface-cli login`.

## Quick start

```python
from vane.extraction import load_model, generate, score_response
from vane.tasks import get_task, render_prompt
from vane import VANEClassifier

model, tok = load_model("Qwen/Qwen2.5-1.5B-Instruct")
task = get_task("gsm8k")

records = []
for item in task.load(max_samples=200):
    out = generate(model, tok, [render_prompt(tok, *task.messages(item))])[0]
    text = tok.decode(out["response_ids"], skip_special_tokens=True)
    rec = score_response(model, out["prompt_ids"], out["response_ids"])   # VANE profiles + log-prob
    rec["is_correct"] = task.score(text, item)[1]
    records.append(rec)

clf = VANEClassifier().fit(records[:150])
p_correct = clf.predict_proba(records[150:])          # higher = more likely correct
```

For hidden states you already have, `vane.compute_profiles(H)` takes `H` of shape `(L, T, d)`:
the outputs of the L transformer blocks for the T generated tokens.

## Running the experiments

```bash
# 1. generation + VANE profiles (GPU, resumable)
python scripts/extract.py --model meta-llama/Meta-Llama-3-8B-Instruct --benchmark gsm8k \
    --output_dir results --name llama-3-8b-instruct
#    Gemma-3-12B-IT: add --attn_implementation eager
#    HumanEval: add --allow_code_execution (runs generated code; use a sandbox)
#    Qwen2.5-72B-Instruct: add --quantize_4bit --batch_size 2

# 2. evaluation (CPU)
python scripts/evaluate.py results/llama-3-8b-instruct_gsm8k --n_jobs 5
```

`evaluate.py` writes `eval.json` with:

| Key | Content | Paper |
|---|---|---|
| `vane` | VANE AUROC (mean +- std over 5 folds) | Table 1 |
| `log_prob` | mean token log-probability AUROC | Table 1 |
| `ablation` | each metric alone and all metrics combined (logistic regression) | Fig. 3a |
| `orthogonality` | VANE + log-probability, Pearson r with log-probability | Fig. 3b, Section 4.3 |
| `selective_prediction` | accuracy at 100%-50% coverage for VANE and log-probability | Fig. 4 |

It also saves `classifier.pkl` (hyperparameters tuned and fit on the whole run), which can score
records of another benchmark of the same model:

```python
from vane import VANEClassifier
from vane.evaluation import load_records
clf = VANEClassifier.load("results/qwen3-14b_gsm8k/classifier.pkl")
p_correct = clf.predict_proba(load_records("results/qwen3-14b_cruxeval"))
```

### Protocol

| Setting | Value |
|---|---|
| Models | Qwen2.5-1.5B-Instruct, Llama-3-8B-Instruct, Ministral-8B-Instruct-2410, Gemma-3-12B-IT, Qwen3-14B (non-thinking mode), Qwen2.5-72B-Instruct (4-bit NF4) |
| Benchmarks | GSM8K test (1,319), MATH-500 (500), HumanEval (164), CRUXEval-O (800), BBH 12-subtask subset (2,865) |
| Decoding | greedy, at most 1,024 new tokens, the model's chat template |
| Hidden states | outputs of the L transformer blocks, generated tokens only |
| Token aggregation | max, mean, and max over the final w = 20 tokens |
| Features | 5 metrics x 3 aggregations, concatenated over all layers (462 dims for 32 layers) |
| Classifier | StandardScaler + scikit-learn `GradientBoostingClassifier` |
| Evaluation | 5-fold stratified cross-validation (seed 42); mean +- std of per-fold AUROC |
| Hyperparameters | Optuna (TPE, 80 trials) over the Gradient Boosting hyperparameters, maximising 5-fold CV AUROC on the training split of each outer fold (nested CV); test folds are never used for tuning |

`records.pkl` holds one dict per sample with the `{prefix}_{max|mean|ans}` per-layer profiles
(`prefix` in `curv, jerk, vel, geodev, tokc`), `mean_log_prob`, `is_correct`, `sample_idx`,
`pred`, `gt` and `gen_length`.

## Notes

* The scoring forward pass runs per sample on the unpadded prompt + response, so the features do
  not depend on the generation batch size. Numbers can differ slightly from the paper because of
  GPU non-determinism and library versions.
* **HumanEval executes model-generated code.** It is disabled unless `--allow_code_execution` is given;
  each program runs in a subprocess with time, memory and file-size limits. Use a sandbox.
* `records.pkl` and `classifier.pkl` are pickle files: only load files you created yourself.
* Generation dominates the cost; the VANE metrics add negligible overhead (under 2 minutes for
  1,319 samples on one A100). Evaluation runs on CPU. Qwen2.5-72B in 4-bit needs a GPU with at
  least 48 GB of memory.

## Citation

```bibtex
@inproceedings{fakharzadehjahromy2026geometric,
  title     = {Geometric Instability of Hidden-State Trajectories Predicts Reasoning Failures in Large Language Models},
  author    = {Fakharzadehjahromy, Hoda and Bueff, Andreas C. and Heintz, Fredrik and Tiger, Mattias and Li, Qing and Li, Jiaqi and Geng, Jiahui},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).
