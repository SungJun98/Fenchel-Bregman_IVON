"""Run the Qwen3-4B FBI training stages or posterior evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MMPO = ROOT / "MMPO"


def build_command(
    mode: str,
    *,
    model: Path,
    train_file: Path,
    output: Path,
    stage: int | None = None,
    checkpoint: Path | None = None,
    eval_file: Path | None = None,
    nodes: int = 2,
    gpus_per_node: int = 4,
    ray_address: str | None = None,
) -> list[str]:
    if mode not in {"train", "eval"} or (mode == "train" and stage not in {1, 2}):
        raise ValueError("choose train stage 1/2 or eval")
    if (mode == "eval" or stage == 2) and checkpoint is None:
        raise ValueError("a checkpoint is required for stage 2 and evaluation")
    if mode == "eval" and eval_file is None:
        raise ValueError("evaluation requires a prepared parquet file")
    if nodes < 1 or gpus_per_node < 1 or nodes * gpus_per_node != 8:
        raise ValueError("the reproduction uses eight GPUs")

    evaluating = mode == "eval"
    data_file = eval_file if evaluating else train_file
    assert data_file is not None
    command = [
        sys.executable,
        "-m",
        "recipe.mmpo.main_mmpo",
        "algorithm.adv_estimator=fb_rloo",
        "algorithm.fb.enabled=True",
        "algorithm.fb.k_train=8",
        "algorithm.fb.failure_ema_beta=0.5",
        f"algorithm.fb.weight_mode={'normalized' if evaluating or stage == 2 else 'lagged'}",
        "algorithm.fb.weight_temperature=1.0",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={train_file}",
        f"data.val_files={data_file}",
        "data.train_batch_size=32",
        "data.val_batch_size=32",
        f"data.max_prompt_length={2048 if evaluating else 1024}",
        "data.max_response_length=4096",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.shuffle=True",
        "data.seed=42",
        f"actor_rollout_ref.model.path={model}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.optimizer=IVON",
        "actor_rollout_ref.actor.optim.optimizer_impl=optim.fbi_ivon",
        "actor_rollout_ref.actor.optim.lr=0.5",
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        "actor_rollout_ref.actor.optim.betas=[0.9,0.999]",
        "actor_rollout_ref.actor.optim.ivon_ess=1e9",
        "actor_rollout_ref.actor.optim.ivon_hess_init=1e-3",
        "actor_rollout_ref.actor.optim.ivon_clip_radius=1e-3",
        "actor_rollout_ref.actor.optim.ivon_state_device=cpu",
        "actor_rollout_ref.actor.optim.ivon_noise_seed=42",
        "actor_rollout_ref.actor.policy_loss.loss_mode=fb_rloo",
        "actor_rollout_ref.actor.loss_scale_factor=4096",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=27648",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.50",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        "actor_rollout_ref.rollout.enable_prefix_caching=True",
        f"actor_rollout_ref.rollout.max_model_len={6144 if evaluating else 5120}",
        "actor_rollout_ref.rollout.max_num_batched_tokens=8192",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "trainer.logger=[console]",
        "trainer.project_name=fbi_reproduction",
        f"trainer.experiment_name=fbi_{mode}{stage or ''}",
        f"trainer.n_gpus_per_node={gpus_per_node}",
        f"trainer.nnodes={nodes}",
        "trainer.total_epochs=20",
        "trainer.total_training_steps=300" if evaluating or stage == 2 else "trainer.total_training_steps=200",
        f"trainer.default_local_dir={output}",
        "trainer.max_actor_ckpt_to_keep=3",
        f"trainer.resume_mode={'resume_path' if checkpoint else 'disable'}",
    ]
    if checkpoint is not None:
        command.append(f"trainer.resume_from_path={checkpoint}")
    if evaluating:
        command.extend(
            (
                "actor_rollout_ref.rollout.prompt_length=2048",
                "actor_rollout_ref.rollout.val_kwargs.n=1",
                "actor_rollout_ref.rollout.val_kwargs.temperature=0.6",
                "actor_rollout_ref.rollout.val_kwargs.top_p=0.95",
                "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
                "algorithm.fb.eval_posterior_draws=16",
                "algorithm.fb.eval_posterior_seed=42000",
                "algorithm.fb.eval_seed=42",
                "trainer.val_before_train=True",
                "trainer.val_only=True",
                "trainer.save_freq=-1",
                "trainer.test_freq=-1",
                f"trainer.validation_data_dir={output / 'posterior_draws'}",
            )
        )
    else:
        command.extend(("trainer.val_before_train=False", "trainer.save_freq=50", "trainer.test_freq=-1"))
    command.append(
        "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="
        + os.pathsep.join((str(ROOT.parent), str(MMPO)))
    )
    if ray_address:
        command.append(f"+ray_kwargs.ray_init.address={ray_address}")
        command.append("+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--stage", type=int, choices=(1, 2))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--ray-address")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("model", "train_file", "eval_file", "manifest", "checkpoint", "output"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    command = build_command(
        args.mode,
        model=args.model,
        train_file=args.train_file,
        output=args.output,
        stage=args.stage,
        checkpoint=args.checkpoint,
        eval_file=args.eval_file,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        ray_address=args.ray_address,
    )
    if args.dry_run:
        print(shlex.join(command))
        return
    if args.nodes > 1 and not args.ray_address:
        parser.error("multi-node execution requires --ray-address")
    if not args.model.is_dir() or not args.train_file.is_file():
        parser.error("model directory and training parquet must exist")
    if args.checkpoint is not None and not (
        (args.checkpoint / "actor").is_dir()
        and (args.checkpoint / "fb_failure_state.pt").is_file()
        and (args.checkpoint / "data.pt").is_file()
    ):
        parser.error("checkpoint must contain actor/, fb_failure_state.pt, and data.pt")
    if args.mode == "eval":
        if args.eval_file is None or not args.eval_file.is_file() or args.manifest is None or not args.manifest.is_file():
            parser.error("evaluation parquet and manifest must exist")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if (manifest.get("problem_count"), manifest.get("candidates_per_problem")) != (1275, 16):
            parser.error("evaluation manifest must describe 1,275 problems and 16 candidates")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("VLLM_USE_V1", "1")
    environment.setdefault("NCCL_CUMEM_ENABLE", "0")
    environment.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "true")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT.parent), str(MMPO), environment.get("PYTHONPATH", ""))
    )
    subprocess.run(command, cwd=MMPO, env=environment, check=True)
    if args.mode == "eval":
        generations = sorted((args.output / "posterior_draws").glob("*/300.jsonl"))
        if len(generations) != 16:
            raise RuntimeError(f"expected 16 posterior draw files, found {len(generations)}")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "aggregate_pass_at_k.py"),
                "--manifest",
                str(args.manifest),
                "--generations",
                *(str(path) for path in generations),
                "--output",
                str(args.output / "metrics.json"),
                "--candidates-output",
                str(args.output / "candidates.jsonl"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
