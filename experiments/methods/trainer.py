import os
import json
import math
import torch
from torch.optim.swa_utils import AveragedModel

from rich.console import Console
from rich.progress import Progress, ProgressColumn, SpinnerColumn, TextColumn
from rich.text import Text

from experiment_logger import ExperimentLogger
from methods.validator import evaluate
from methods.losses import (
    base_weighted_pearson_loss,
    focal_weighted_pearson_loss,
    mse_anchor_loss,
    dynamic_trimmed_loss,
    asym_corr_penalty_loss,
    local_global_weighted_pearson_loss,
)

import methods.base_method as base_method
import methods.cosine_scheduler_method as cosine_scheduler_method
import methods.focal_wp_method as focal_wp_method
import methods.mse_anchor as mse_anchor
import methods.swa_method as swa_method
import methods.local_global_loss_7_3 as local_global_loss_7_3
import methods.dynamic_trimmed_method as dynamic_trimmed_method
import methods.asym_wp_loss as asym_wp_loss


class UVDashedBarColumn(ProgressColumn):
    def __init__(self, bar_width: int = 24):
        super().__init__()
        self.bar_width = bar_width

    def render(self, task):
        if task.fields.get("is_header"):
            return Text("")

        if not task.total:
            return Text("-" * self.bar_width, style="#2e3440")

        ratio = min(1.0, max(0.0, task.completed / task.total))
        filled = int(self.bar_width * ratio)
        unfilled = self.bar_width - filled

        text = Text()
        text.append("-" * filled, style="green")
        text.append("-" * unfilled, style="#2e3440")
        return text


class UVNameOrSpinnerColumn(ProgressColumn):
    def __init__(self, name_width: int = 34):
        super().__init__()
        self.name_width = name_width
        self.spinner = SpinnerColumn(spinner_name="dots", style="green").spinner

    def render(self, task):
        if task.fields.get("is_header"):
            spinner_char = self.spinner.render(task.get_time())
            title = task.fields.get("title", "Training...")
            return Text.assemble(spinner_char, Text(f" {title}", style="bold white"))

        name = task.description
        if len(name) > self.name_width:
            name = name[: self.name_width - 3] + "..."
        return Text(f"{name:<{self.name_width}}", style="bold cyan")


class UVStepCounterColumn(ProgressColumn):
    def render(self, task):
        if task.fields.get("is_header"):
            return Text("")
        total = task.total if task.total else 0
        return Text(f"{int(task.completed):04d}/{int(total):04d}", style="dim white")


class UVMetricsColumn(ProgressColumn):
    def render(self, task):
        if task.fields.get("is_header"):
            return Text("")

        if not task.fields.get("active", True):
            best_wp = task.fields.get("best_wp", 0.0)
            return Text(f"[STOPPED] Best WP: {best_wp:.4f}", style="dim red")

        loss = task.fields.get("loss", 0.0)
        best_wp = task.fields.get("best_wp", -1.0)
        lr = task.fields.get("lr", 0.0)
        patience = task.fields.get("patience", 0)
        p_limit = task.fields.get("p_limit", 100)
        b_size = task.fields.get("b_size", 0)

        best_str = f"{best_wp:+.4f}" if best_wp != -float("inf") else " N/A  "
        return Text(
            f"B:{b_size:02d} | Loss: {loss:+.4f} | Best WP: {best_str} | Pat: {patience:02d}/{p_limit:02d} | LR: {lr:.1e}",
            style="white",
        )


class SlotRuntime:
    def __init__(self, exp_dict, cfg, device, total_stream_steps, base_batch_size):
        self.exp_dict = exp_dict
        self.name = exp_dict["name"]
        self.model_module = exp_dict["model"]
        self.method_module = exp_dict["method"]

        self.target_batch_size = exp_dict.get("batch_size", base_batch_size)
        self.accum_steps = max(1, self.target_batch_size // base_batch_size)

        self.model = self.model_module.create_model(cfg).to(device)

        self.lr = exp_dict.get("lr", cfg.lr)
        self.weight_decay = exp_dict.get("weight_decay", cfg.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.loss_fn, self.scheduler_type, self.loss_step_hook, self.use_swa = self._resolve_method(cfg)

        self.scheduler = None
        effective_opt_steps = max(1, total_stream_steps // self.accum_steps)
        if self.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=effective_opt_steps, eta_min=getattr(cfg, "min_lr", 1e-6)
            )

        self.swa_model = AveragedModel(self.model) if self.use_swa else None
        self.swa_start_epoch = exp_dict.get("swa_start_epoch", 2)
        self.patience_limit = exp_dict.get("patience_limit", 100)

        self.best_sub_wp = -float("inf")
        self.best_swa_sub_wp = -float("inf")
        self.patience = 0
        self.is_active = True

        self.step_loss_acc = 0.0
        self.step_wp_acc = 0.0
        self.step_count = 0
        self.last_avg_loss = 0.0

    def _resolve_method(self, cfg):
        method = self.method_module
        exp = self.exp_dict

        if method == base_method:
            return base_weighted_pearson_loss, None, None, False
        elif method == cosine_scheduler_method:
            return base_weighted_pearson_loss, "cosine", None, False
        elif method == focal_wp_method:
            gamma = exp.get("focal_gamma", getattr(cfg, "focal_gamma", 2.0))
            loss = lambda p, y: focal_weighted_pearson_loss(p, y, gamma=gamma)
            return loss, "cosine", None, False
        elif method == mse_anchor:
            lambda_init = exp.get("lambda_anchor", getattr(cfg, "lambda_anchor", 0.05))
            hook = lambda s, tot, c: {"lambda_anchor": 0.5 * lambda_init * (1.0 + math.cos(math.pi * min(1.0, s / tot)))}
            return mse_anchor_loss, "cosine", hook, False
        elif method == swa_method:
            return base_weighted_pearson_loss, None, None, True
        elif method == local_global_loss_7_3:
            return local_global_weighted_pearson_loss, None, None, False
        elif method == dynamic_trimmed_method:
            return (lambda p, y: dynamic_trimmed_loss(p, y, keep_ratio=0.70, soft_weight=0.05)), "cosine", None, False
        elif method == asym_wp_loss:
            return asym_corr_penalty_loss, None, None, False

        return base_weighted_pearson_loss, None, None, False


def train_seed(experiments, train_loader, cfg, exp_name=None, seed=None):
    if isinstance(experiments, dict):
        exp_list = [experiments]
    elif isinstance(experiments, (list, tuple)):
        exp_list = list(experiments)
    else:
        raise ValueError("experiments must be dict or list of dicts")

    seed_val = seed if seed is not None else cfg.seeds[0]
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    base_batch_size = getattr(cfg, "full_batch_size", 4)
    num_batches = len(train_loader)
    total_steps = cfg.max_epochs * num_batches

    slots = [SlotRuntime(exp_dict, cfg, device, total_steps, base_batch_size) for exp_dict in exp_list]
    loggers = {s.name: ExperimentLogger(cfg.runs_dir, s.name, seed_val) for s in slots}

    val_interval = max(1, num_batches // 5)
    val_steps_in_epoch = set([val_interval * k for k in range(1, 5)])
    val_steps_in_epoch.add(num_batches)

    global_step = 0
    console = Console()

    progress = Progress(
        UVNameOrSpinnerColumn(name_width=32),
        UVDashedBarColumn(bar_width=20),
        TextColumn(" "),
        UVStepCounterColumn(),
        TextColumn(" "),
        UVMetricsColumn(),
        console=console,
        transient=False,
    )

    with progress:
        header_task = progress.add_task(
            "",
            is_header=True,
            title=f"Starting training on Seed {seed_val} | Stream batches: {num_batches} | Steps: {total_steps}",
        )

        tasks = {
            s.name: progress.add_task(
                s.name,
                total=total_steps,
                is_header=False,
                active=True,
                loss=0.0,
                best_wp=s.best_sub_wp,
                lr=s.lr,
                patience=0,
                p_limit=s.patience_limit,
                b_size=s.target_batch_size,
            )
            for s in slots
        }

        for epoch in range(1, cfg.max_epochs + 1):
            for s in slots:
                if s.is_active:
                    s.model.train()

            for step_in_epoch, (x, y) in enumerate(train_loader, 1):
                global_step += 1
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                active_slots = [s for s in slots if s.is_active]
                if not active_slots:
                    break

                for s in active_slots:
                    extra_kwargs = {}
                    if s.loss_step_hook is not None:
                        extra_kwargs = s.loss_step_hook(global_step, total_steps, cfg)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = s.model(x)
                        preds = out[0] if isinstance(out, tuple) else out
                        loss = s.loss_fn(preds, y, **extra_kwargs)
                        scaled_loss = loss / s.accum_steps

                    scaled_loss.backward()

                    if global_step % s.accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(s.model.parameters(), max_norm=1.0)
                        s.optimizer.step()
                        s.optimizer.zero_grad(set_to_none=True)

                        if s.scheduler is not None:
                            s.scheduler.step()

                        if s.use_swa and epoch >= s.swa_start_epoch:
                            s.swa_model.update_parameters(s.model)

                    step_loss = loss.item()
                    s.step_loss_acc += step_loss
                    s.step_wp_acc += (-step_loss)
                    s.step_count += 1

                    if global_step % cfg.log_interval == 0:
                        s.last_avg_loss = s.step_loss_acc / s.step_count
                        avg_train_wp = s.step_wp_acc / s.step_count
                        current_lr = s.optimizer.param_groups[0]["lr"]
                        loggers[s.name].log_train_step(epoch, global_step, current_lr, s.last_avg_loss, avg_train_wp)
                        s.step_loss_acc, s.step_wp_acc, s.step_count = 0.0, 0.0, 0

                    progress.update(
                        tasks[s.name],
                        completed=global_step,
                        loss=s.last_avg_loss,
                        best_wp=s.best_sub_wp,
                        lr=s.optimizer.param_groups[0]["lr"],
                        patience=s.patience,
                        active=s.is_active,
                    )

                progress.update(
                    header_task,
                    title=f"Epoch {epoch:02d}/{cfg.max_epochs:02d} | Step {step_in_epoch:04d}/{num_batches:04d} (Global {global_step:05d})",
                )

                if step_in_epoch in val_steps_in_epoch:
                    models_to_eval = {s.name: s.model for s in slots if s.is_active}
                    if models_to_eval:
                        sub_results = evaluate(models_to_eval, cfg, device, sample_stride=10)

                        for s in [s for s in slots if s.is_active]:
                            res = sub_results[s.name]
                            sub_wp = res["weighted_pearson"]
                            logger = loggers[s.name]
                            ckpt_path = os.path.join(logger.run_dir, "best_sub_model.pt")

                            if sub_wp > s.best_sub_wp:
                                s.best_sub_wp = sub_wp
                                s.patience = 0
                                torch.save(s.model.state_dict(), ckpt_path)
                                mark = " [bold green][NEW BEST!][/bold green]"
                            else:
                                s.patience += 1
                                mark = f" [dim](No change: {s.patience}/{s.patience_limit})[/dim]"

                            current_lr = s.optimizer.param_groups[0]["lr"]
                            progress.console.print(
                                f"  [dim]↳[/dim] [cyan]{s.name:<32}[/cyan] | Ep {epoch:02d} | "
                                f"WP: [bold white]{sub_wp:.4f}[/bold white] (t0: {res['t0']:.4f}, t1: {res['t1']:.4f}){mark}"
                            )
                            logger.log_val_event(epoch, global_step, "sub_10pct", current_lr, res, s.patience)

                            if s.patience >= s.patience_limit:
                                progress.console.print(
                                    f"  [red]↳ Early stopping triggered for {s.name} at step {global_step}.[/red]"
                                )
                                s.is_active = False

                            progress.update(
                                tasks[s.name],
                                best_wp=s.best_sub_wp,
                                patience=s.patience,
                                active=s.is_active,
                            )

                    for s in slots:
                        if s.is_active:
                            s.model.train()

            if not any(s.is_active for s in slots):
                break

    console.print("\n[bold yellow]>>> Evaluating best checkpoints on 100% validation dataset...[/bold yellow]")
    best_models = {}
    for s in slots:
        logger = loggers[s.name]
        ckpt_path = os.path.join(logger.run_dir, "best_sub_model.pt")
        if os.path.exists(ckpt_path):
            s.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        best_models[s.name] = s.model

    final_results = evaluate(best_models, cfg, device, sample_stride=1)

    final_scores = {}
    for s in slots:
        res = final_results[s.name]
        wp = res["weighted_pearson"]
        final_scores[s.name] = wp
        logger = loggers[s.name]
        console.print(
            f"  [bold green]✓[/bold green] [cyan]{s.name:<32}[/cyan] -> "
            f"FULL VAL WP: [bold white]{wp:.5f}[/bold white] (t0: {res['t0']:.4f}, t1: {res['t1']:.4f})"
        )

        summary_data = {
            "seed": seed_val,
            "best_sub_wp": s.best_sub_wp,
            "final_full_wp": wp,
            "epochs_trained": epoch,
            "total_steps": global_step,
        }
        with open(os.path.join(logger.run_dir, "seed_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)

    return final_scores if len(exp_list) > 1 else final_scores[slots[0].name]