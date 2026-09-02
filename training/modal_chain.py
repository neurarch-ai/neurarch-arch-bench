#!/usr/bin/env python3
"""Run the Arch-Bench training chain on Modal GPUs.

The chain is the one in GRPO_RUN_GUIDE.md — mint verified pairs, evaluate the
untrained policy, SFT, evaluate, GRPO on top, evaluate — but on rented A10G /
A100 instead of a free Colab T4. Two things that guide leaves to the operator
are automated here: the env-server (the reward) runs inside the same container
as the trainer, and every stage's metrics are parsed out of stdout and returned
as JSON so a run is a table, not a scrollback.

  modal run training/modal_chain.py                      # 1.5B chain
  modal run training/modal_chain.py --model Qwen/Qwen2.5-7B-Instruct --gpu A100-80GB
  modal run training/modal_chain.py --stage eval --model-path /runs/...

Artifacts (checkpoints, minted data, per-stage JSON) live in the `archbench-runs`
volume under /runs/<tag>/; the HF cache is its own volume so a re-run does not
re-download the base model.
"""
import json
import os
import re
import subprocess
import time

import modal

app = modal.App("neurarch-archbench-train")

# Torch first (its own index), then the training stack pinned to the era the
# scripts were written against — TRL's GRPOConfig/SFTConfig fields move between
# minor versions, and both trainers only defend against drift, they cannot
# invent a missing arg. node is for env-server.mjs (the reward) and the
# dataset minter; both are zero-dependency, so debian's node 18 is enough.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("nodejs", "npm", "git")
    .pip_install(
        "torch==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.48.3",
        "trl==0.14.0",
        "peft==0.14.0",
        "accelerate==1.3.0",
        "datasets==3.2.0",
        "numpy<2",
    )
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        remote_path="/bench",
        ignore=["**/node_modules/**", "**/.git/**", "**/out/**", "**/__pycache__/**"],
    )
)

runs = modal.Volume.from_name("archbench-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("archbench-hf", create_if_missing=True)
VOLUMES = {"/runs": runs, "/hf": hf_cache}

ENV_URL = "http://127.0.0.1:8737"


# ── helpers that run inside the container ────────────────────────────────────

def start_env_server():
    """Boot env-server.mjs and block until /health answers."""
    import urllib.request

    proc = subprocess.Popen(
        ["node", "env-server.mjs"],
        cwd="/bench",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{ENV_URL}/health", timeout=2) as r:
                if json.loads(r.read()).get("ok"):
                    print("[env] reward server up")
                    return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("env-server did not become healthy")


def run(cmd, cwd="/bench"):
    """Run a stage, stream its output, return the whole transcript."""
    print(f"[run] {' '.join(cmd)}", flush=True)
    lines = []
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        lines.append(line)
        print(line.rstrip(), flush=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"stage failed ({p.returncode}): {' '.join(cmd)}")
    return "".join(lines)


def parse_eval(out: str) -> dict:
    """Pull the three numbers train_grpo.py --eval-only prints at the end."""
    m = re.search(r"pass@1: (\d+)/(\d+) = ([\d.]+)", out)
    pf = re.search(r"parse failures: (\d+)/(\d+)", out)
    mr = re.search(r"mean reward: (-?[\d.]+)", out)
    if not m:
        raise RuntimeError("no pass@1 line in eval output")
    return {
        "passed": int(m.group(1)),
        "n": int(m.group(2)),
        "pass_at_1": float(m.group(3)),
        "parse_failures": int(pf.group(1)) if pf else None,
        "mean_reward": float(mr.group(1)) if mr else None,
    }


def rubric_version() -> int:
    out = run(["node", "-e",
               "import('./bench.mjs').then(m => console.log(m.RUBRIC_VERSION))"])
    return int(out.strip().splitlines()[-1])


def save(tag: str, name: str, payload: dict):
    d = f"/runs/{tag}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/{name}.json", "w") as f:
        json.dump(payload, f, indent=2)
    runs.commit()


# ── stages ───────────────────────────────────────────────────────────────────

@app.function(image=image, volumes=VOLUMES, timeout=1800)
def mint(count: int = 3000, seed: int = 20260716, tag: str = "shared"):
    """Mint verified (spec -> actions) pairs, holdout-excluded, keylessly."""
    os.makedirs(f"/runs/{tag}/data", exist_ok=True)
    out = f"/runs/{tag}/data/sft-{count}"
    log = run(["node", "training/build_sft_dataset.mjs",
               f"--count={count}", f"--seed={seed}", f"--out={out}"])
    rows = sum(1 for _ in open(f"{out}.chat.jsonl"))
    excluded = re.search(r"excluded (\d+) row", log)
    payload = {
        "path": f"{out}.chat.jsonl",
        "rows": rows,
        "nominal": count,
        "seed": seed,
        "excluded_holdout_collisions": int(excluded.group(1)) if excluded else 0,
        "rubric_version": rubric_version(),
    }
    save(tag, f"mint-{count}", payload)
    print(json.dumps(payload, indent=2))
    return payload


# Evaluation is a 1.5B forward pass; L4 and T4 schedule far more readily than
# A10G when the region is tight, and the eval is not throughput-bound.
# Evaluation is a 1.5B forward pass, so it is indifferent to GPU class and to
# cloud. Naming several of both is the difference between scheduling in
# minutes and sitting in a queue for a day when one region is tight.
@app.function(image=image, gpu=["L4", "T4", "A10G"], cloud="auto",
              volumes=VOLUMES, timeout=14400)
def evaluate(model: str, tag: str, label: str, seed: int = 999,
             count: int = 192, curated: bool = False, max_completion: int = 384,
             families: str = ""):
    """pass@1 on a strictly-held-out split (or the 12 curated tasks)."""
    proc = start_env_server()
    try:
        cmd = ["python", "-u", "training/train_grpo.py", "--eval-only",
               "--model", model, "--env-url", ENV_URL,
               "--max-completion", str(max_completion)]
        cmd += ["--curated"] if curated else ["--seed", str(seed), "--count", str(count)]
        if families:
            cmd += ["--families", families]
        res = parse_eval(run(cmd))
        res |= {"model": model, "label": label, "curated": curated,
                "seed": None if curated else seed, "families": families or None,
                "rubric_version": rubric_version()}
        save(tag, f"eval-{label}", res)
        print(json.dumps(res, indent=2))
        return res
    finally:
        proc.terminate()


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=14400)
def sft(data: str, tag: str, model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        epochs: float = 2.0, batch_size: int = 4, grad_accum: int = 4):
    out = f"/runs/{tag}/sft"
    run(["python", "-u", "training/train_sft.py", "--data", data, "--model", model,
         "--out", out, "--epochs", str(epochs), "--bf16",
         "--batch-size", str(batch_size), "--grad-accum", str(grad_accum)])
    runs.commit()
    ckpt = f"{out}/checkpoint-final"
    save(tag, "sft", {"checkpoint": ckpt, "base_model": model,
                      "data": data, "epochs": epochs})
    return ckpt


@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=21600)
def grpo(model: str, tag: str, steps: int = 100, lr: float = 1e-5,
         count: int = 256, seed: int = 123, batch_size: int = 4,
         num_generations: int = 4, max_completion: int = 384, suffix: str = ""):
    """GRPO against the verifier's reward, starting from an SFT checkpoint."""
    proc = start_env_server()
    out = f"/runs/{tag}/grpo{suffix}"
    try:
        run(["python", "-u", "training/train_grpo.py", "--model", model,
             "--out", out, "--steps", str(steps), "--lr", str(lr),
             "--count", str(count), "--seed", str(seed), "--lora", "--bf16",
             "--batch-size", str(batch_size),
             "--num-generations", str(num_generations),
             "--max-completion", str(max_completion), "--env-url", ENV_URL])
    finally:
        proc.terminate()
    runs.commit()
    # The reward curve is figure material; keep it next to the metrics.
    curve = []
    state = f"{out}/checkpoint-final/trainer_state.json"
    if not os.path.exists(state):
        state = f"{out}/trainer_state.json"
    if os.path.exists(state):
        hist = json.load(open(state)).get("log_history", [])
        curve = [{"step": h.get("step"), "reward": h.get("reward")}
                 for h in hist if h.get("reward") is not None]
    ckpt = f"{out}/checkpoint-final"
    save(tag, f"grpo{suffix}", {"checkpoint": ckpt, "from": model, "steps": steps,
                                "lr": lr, "train_seed": seed, "reward_curve": curve})
    return ckpt


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct",
         gpu: str = "A10G",
         tag: str = "",
         data_count: int = 3000,
         data_seed: int = 20260716,
         eval_count: int = 192,
         eval_seed: int = 999,
         grpo_steps: int = 100,
         grpo_lr: float = 1e-5,
         grpo_count: int = 256,
         batch_size: int = 4,
         grad_accum: int = 4,
         skip_curated: bool = False,
         stages: str = "mint,base,sft,sft_eval,grpo,grpo_eval"):
    """Run the full chain and print the paper table."""
    tag = tag or model.split("/")[-1].lower()
    want = {s.strip() for s in stages.split(",") if s.strip()}
    gpu_opts = {"gpu": gpu} if gpu else {}
    _eval = evaluate.with_options(**gpu_opts)
    _sft = sft.with_options(**gpu_opts)
    _grpo = grpo.with_options(**gpu_opts)

    results = {"tag": tag, "model": model, "gpu": gpu}

    data = f"/runs/shared/data/sft-{data_count}.chat.jsonl"
    if "mint" in want:
        results["data"] = mint.remote(count=data_count, seed=data_seed, tag="shared")
        data = results["data"]["path"]

    if "base" in want:
        results["untrained"] = _eval.remote(
            model=model, tag=tag, label="untrained",
            seed=eval_seed, count=eval_count)
        if not skip_curated:
            results["untrained_curated"] = _eval.remote(
                model=model, tag=tag, label="untrained-curated", curated=True)

    sft_ckpt = f"/runs/{tag}/sft/checkpoint-final"
    if "sft" in want:
        sft_ckpt = _sft.remote(data=data, tag=tag, model=model,
                               batch_size=batch_size, grad_accum=grad_accum)
    if "sft_eval" in want:
        results["sft"] = _eval.remote(model=sft_ckpt, tag=tag, label="sft",
                                      seed=eval_seed, count=eval_count)
        if not skip_curated:
            results["sft_curated"] = _eval.remote(
                model=sft_ckpt, tag=tag, label="sft-curated", curated=True)

    grpo_ckpt = f"/runs/{tag}/grpo/checkpoint-final"
    if "grpo" in want:
        grpo_ckpt = _grpo.remote(model=sft_ckpt, tag=tag, steps=grpo_steps,
                                 lr=grpo_lr, count=grpo_count,
                                 batch_size=batch_size)
    if "grpo_eval" in want:
        results["grpo"] = _eval.remote(model=grpo_ckpt, tag=tag, label="grpo",
                                       seed=eval_seed, count=eval_count)
        if not skip_curated:
            results["grpo_curated"] = _eval.remote(
                model=grpo_ckpt, tag=tag, label="grpo-curated", curated=True)

    print("\n" + json.dumps(results, indent=2))
    print(f"\n{'stage':<12}{'pass@1':>12}{'parse fail':>14}{'mean reward':>14}")
    for key, name in [("untrained", "untrained"), ("sft", "+SFT"), ("grpo", "+GRPO")]:
        r = results.get(key)
        if r:
            print(f"{name:<12}{r['passed']}/{r['n']} = {r['pass_at_1']:.3f}"
                  f"{str(r['parse_failures']) + '/' + str(r['n']):>14}"
                  f"{r['mean_reward']:>14.3f}")


# ── one-shot remote chain ────────────────────────────────────────────────────
# A local entrypoint orchestrating .remote() calls dies with the local client
# (--detach only protects the last triggered function), and a dropped laptop
# connection then cancels a multi-hour run mid-stage. This runs every stage
# inside one container instead, so the only thing the local CLI does is start
# it. Results land in the volume stage by stage, so a crash loses one stage.

@app.function(image=image, gpu="A100-80GB", volumes=VOLUMES, timeout=86400)
def chain(model: str, tag: str, data: str, eval_seed: int = 999,
          eval_count: int = 192, grpo_steps: int = 100, grpo_lr: float = 1e-5,
          grpo_count: int = 256, batch_size: int = 4, grad_accum: int = 4,
          max_completion: int = 384, stages: str = "base,sft,sft_eval,grpo,grpo_eval",
          skip_curated: bool = False,
          dtype: str = ""):
    want = {s.strip() for s in stages.split(",") if s.strip()}
    results = {"tag": tag, "model": model, "stages": sorted(want)}
    proc = start_env_server()
    rubric = rubric_version()

    def _eval(model_path, label, curated=False):
        cmd = ["python", "-u", "training/train_grpo.py", "--eval-only",
               "--model", model_path, "--env-url", ENV_URL,
               "--max-completion", str(max_completion)]
        cmd += ["--curated"] if curated else ["--seed", str(eval_seed),
                                              "--count", str(eval_count)]
        res = parse_eval(run(cmd))
        res |= {"model": model_path, "label": label, "curated": curated,
                "seed": None if curated else eval_seed, "rubric_version": rubric}
        save(tag, f"eval-{label}", res)
        results[label] = res
        print(f"[stage] {label}: {json.dumps(res)}", flush=True)
        return res

    try:
        if "base" in want:
            _eval(model, "untrained")
            if not skip_curated:
                _eval(model, "untrained-curated", curated=True)

        sft_ckpt = f"/runs/{tag}/sft/checkpoint-final"
        if "sft" in want:
            out = f"/runs/{tag}/sft"
            run(["python", "-u", "training/train_sft.py", "--data", data,
                 "--model", model, "--out", out, "--epochs", "2.0", "--bf16",
                 "--batch-size", str(batch_size), "--grad-accum", str(grad_accum)])
            runs.commit()
            save(tag, "sft", {"checkpoint": sft_ckpt, "base_model": model, "data": data})
        if "sft_eval" in want:
            _eval(sft_ckpt, "sft")
            if not skip_curated:
                _eval(sft_ckpt, "sft-curated", curated=True)

        grpo_out = f"/runs/{tag}/grpo"
        grpo_ckpt = f"{grpo_out}/checkpoint-final"
        if "grpo" in want:
            run(["python", "-u", "training/train_grpo.py", "--model", sft_ckpt,
                 "--out", grpo_out, "--steps", str(grpo_steps), "--lr", str(grpo_lr),
                 "--count", str(grpo_count), "--seed", "123", "--lora", "--bf16",
                 "--batch-size", str(batch_size), "--num-generations", "4",
                 "--max-completion", str(max_completion), "--env-url", ENV_URL]
                + (["--dtype", dtype] if dtype else []))
            runs.commit()
            curve = []
            for state in (f"{grpo_ckpt}/trainer_state.json", f"{grpo_out}/trainer_state.json"):
                if os.path.exists(state):
                    curve = [{"step": h.get("step"), "reward": h.get("reward")}
                             for h in json.load(open(state)).get("log_history", [])
                             if h.get("reward") is not None]
                    break
            save(tag, "grpo", {"checkpoint": grpo_ckpt, "from": sft_ckpt,
                               "steps": grpo_steps, "lr": grpo_lr, "reward_curve": curve})
        if "grpo_eval" in want:
            _eval(grpo_ckpt, "grpo")
            if not skip_curated:
                _eval(grpo_ckpt, "grpo-curated", curated=True)
    finally:
        proc.terminate()

    save(tag, "chain-summary", results)
    print(json.dumps(results, indent=2))
    return results


# ── grounding at scale ───────────────────────────────────────────────────────
# Building and training a few hundred small graphs is embarrassingly parallel
# and CPU-bound locally (2.5 hours for 780 graphs on a laptop); one GPU does it
# while the training chain is running in another container.

@app.function(image=image, gpu="A10G", volumes=VOLUMES, timeout=21600)
def grounding(count: int = 200, seed: int = 4242, steps: int = 800,
              clean_only: bool = True, tag: str = "grounding", label: str = "clean",
              shard: int = 0, shards: int = 1):
    """Mint (architecture, verdict, real training curve) triples.

    Training 800 steps per graph costs ~2.7 minutes, so 200 graphs is nine hours
    in one container and an hour across eight. The set is dumped once, then each
    shard trains its own stride of it and writes its own file; nothing is shared
    but the volume.
    """
    os.makedirs(f"/runs/{tag}", exist_ok=True)
    jsonl = f"/runs/{tag}/{label}{count}.jsonl"
    if shard == 0 or not os.path.exists(jsonl):
        cmd = ["node", "training/dump_grounding_set.mjs", f"--count={count}",
               f"--seed={seed}", f"--out={jsonl}"]
        if clean_only:
            cmd.append("--clean-only")
        run(cmd)
        runs.commit()
    else:
        for _ in range(60):
            runs.reload()
            if os.path.exists(jsonl):
                break
            time.sleep(2)

    rows = [l for l in open(jsonl) if l.strip()]
    mine = rows[shard::shards]
    part = f"/runs/{tag}/{label}{count}-shard{shard}.jsonl"
    with open(part, "w") as f:
        f.writelines(mine)
    print(f"shard {shard}/{shards}: {len(mine)} of {len(rows)} graphs", flush=True)

    out = f"/runs/{tag}/{label}{count}-triples-shard{shard}.jsonl"
    run(["python", "-u", "training/grounding_at_scale.py", "--set", part,
         "--steps", str(steps), "--out", out])
    runs.commit()
    n = sum(1 for _ in open(out))
    print(f"shard {shard}: wrote {n} triples")
    return {"shard": shard, "triples": out, "rows": n}


# ── family-holdout transfer ──────────────────────────────────────────────────
# The paper's transfer evidence is twelve curated tasks, which it calls
# suggestive and which is the right word for n=12. The generator makes a proper
# out-of-distribution split free: fine-tune on six families, evaluate on the
# four the model has never seen.

@app.function(image=image, gpu="A100-40GB", volumes=VOLUMES, timeout=28800)
def family_holdout(heldout: str = "txf,gqa,tower,norm", tag: str = "ood",
                   model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                   mint_count: int = 6000, mint_seed: int = 20260716,
                   eval_count: int = 512, eval_seed: int = 999,
                   batch_size: int = 4, grad_accum: int = 4):
    proc = start_env_server()
    rubric = rubric_version()
    results = {"tag": tag, "heldout": heldout, "model": model, "rubric_version": rubric}
    try:
        # Mint from the six training families only.
        os.makedirs(f"/runs/{tag}/data", exist_ok=True)
        data_stem = f"/runs/{tag}/data/sft-{mint_count}-minus-{heldout.replace(',', '_')}"
        log = run(["node", "training/build_sft_dataset.mjs", f"--count={mint_count}",
                   f"--seed={mint_seed}", f"--exclude-families={heldout}",
                   f"--out={data_stem}"])
        data = f"{data_stem}.chat.jsonl"
        results["train_rows"] = sum(1 for _ in open(data))
        results["train_families_excluded"] = heldout

        def _eval(model_path, label, families=None):
            cmd = ["python", "-u", "training/train_grpo.py", "--eval-only",
                   "--model", model_path, "--env-url", ENV_URL,
                   "--seed", str(eval_seed), "--count", str(eval_count),
                   "--max-completion", "384"]
            if families:
                cmd += ["--families", families]
            res = parse_eval(run(cmd))
            res |= {"model": model_path, "label": label, "families": families,
                    "rubric_version": rubric}
            save(tag, f"eval-{label}", res)
            results[label] = res
            print(f"[stage] {label}: {json.dumps(res)}", flush=True)

        # Untrained baseline on the held-out families, then on the seen ones.
        _eval(model, "untrained-heldout", families=heldout)
        _eval(model, "untrained-seen", families="mlp,ae,cnn,fix,trim,grow")

        out = f"/runs/{tag}/sft"
        run(["python", "-u", "training/train_sft.py", "--data", data, "--model", model,
             "--out", out, "--epochs", "2.0", "--bf16",
             "--batch-size", str(batch_size), "--grad-accum", str(grad_accum)])
        runs.commit()
        ckpt = f"{out}/checkpoint-final"

        # The measurement: does training on six families transfer to four unseen
        # ones, and by how much less than it transfers within distribution?
        _eval(ckpt, "sft-heldout", families=heldout)
        _eval(ckpt, "sft-seen", families="mlp,ae,cnn,fix,trim,grow")
    finally:
        proc.terminate()
    save(tag, "summary", results)
    print(json.dumps(results, indent=2))
    return results
