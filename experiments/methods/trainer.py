import copy
import json
import math
import os
import time
import torch
from rich.console import Console
from rich.progress import Progress, ProgressColumn, SpinnerColumn, TextColumn
from rich.text import Text
from torch.optim.swa_utils import AveragedModel

from experiment_logger import ExperimentLogger
from methods.losses import (
    asym_corr_penalty_loss,
    base_weighted_pearson_loss,
    dynamic_trimmed_loss,
    focal_weighted_pearson_loss,
    local_global_weighted_pearson_loss,
    mse_anchor_loss,
)
from methods.validator import evaluate


def format_time(seconds: float) -> str:
    """Форматирует секунды в человекочитаемый вид MM:SS или HH:MM:SS."""
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_val_history(history, best_wp, last_res=None, is_active=True, max_items=4):
    """Формирует одну компактную строку с динамикой/хронологией метрик."""
    if not history:
        return Text("Waiting for first evaluation...", style="dim italic")

    t = Text()
    visible = history[-max_items:]
    if len(history) > max_items:
        t.append("… → ", style="dim")

    for i, item in enumerate(visible):
        wp = item["wp"]
        diff = item["diff"]
        is_best = item["is_best"]
        wp_str = f"{wp:+.4f}"

        if is_best:
            t.append(wp_str, style="bold green")
            t.append("★", style="bold yellow")
        elif diff > 0:
            t.append(wp_str, style="green")
            t.append("↑", style="green")
        elif diff < 0:
            t.append(wp_str, style="dim red")
            t.append("↓", style="dim red")
        else:
            t.append(wp_str, style="dim white")

        if i < len(visible) - 1:
            t.append(" → ", style="dim")

    if last_res:
        t.append(f" | t0: {last_res['t0']:.3f} t1: {last_res['t1']:.3f}", style="dim")

    best_str = f"{best_wp:+.4f}" if best_wp != -float("inf") else "N/A"
    t.append(f" | Best: {best_str}", style="bold white")

    if not is_active:
        t.append(" [STOPPED]", style="bold red")

    return t


class UVDashedBarColumn(ProgressColumn):
    def __init__(self, bar_width: int = 24):
        super().__init__()
        self.bar_width = bar_width

    def render(self, task):
        if task.fields.get("is_header") or task.fields.get("is_subval"):
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

        if task.fields.get("is_subval"):
            sub_title = "  ↳ Sub-Val Dynamics:"
            return Text(f"{sub_title:<{self.name_width}}", style="dim cyan")

        name = task.description
        if len(name) > self.name_width:
            name = name[: self.name_width - 3] + "..."
        return Text(f"{name:<{self.name_width}}", style="bold cyan")


class UVStepCounterColumn(ProgressColumn):
    def render(self, task):
        if task.fields.get("is_header") or task.fields.get("is_subval"):
            return Text("")
        total = task.total if task.total else 0
        return Text(f"{int(task.completed):04d}/{int(total):04d}", style="dim white")


class UVMetricsColumn(ProgressColumn):
    def render(self, task):
        if task.fields.get("is_header"):
            return Text("")

        if task.fields.get("is_subval"):
            return task.fields.get("val_line", Text("Waiting for first eval...", style="dim italic"))

        if not task.fields.get("active", True):
            best_wp = task.fields.get("best_wp", 0.0)
            return Text(f"[STOPPED] Best WP: {best_wp:.4f}", style="dim red")

        loss = task.fields.get("loss", 0.0)
        best_wp = task.fields.get("best_wp", -1.0)
        lr = task.fields.get("lr", 0.0)

        best_str = f"{best_wp:+.4f}" if best_wp != -float("inf") else " N/A  "
        # Убраны B:XX и Pat: XX/XX
        return Text(
            f"Loss: {loss:+.4f} | Best WP: {best_str} | LR: {lr:.1e}",
            style="white",
        )


class SlotRuntime:
    def __init__(self, exp_dict, cfg, device, total_steps, base_batch_size=None):
        self.exp_dict = exp_dict
        self.name = exp_dict["name"]
        self.model_module = exp_dict["model"]
        self.method_module = exp_dict["method"]

        slot_cfg = copy.copy(cfg)
        for key, value in exp_dict.items():
            if hasattr(slot_cfg, key):
                setattr(slot_cfg, key, value)

        fallback_batch = base_batch_size if base_batch_size is not None else getattr(slot_cfg, "full_batch_size", 5)
        self.target_batch_size = exp_dict.get("batch_size", fallback_batch)
        self.accum_steps = exp_dict.get("accum_steps", getattr(slot_cfg, "accum_steps", 1))

        chunks_per_seq = 20_000 // getattr(slot_cfg, "chunk_size", 2000)
        self.num_chunks = self.target_batch_size * chunks_per_seq

        self.model = self.model_module.create_model(slot_cfg).to(device)

        self.lr = exp_dict.get("lr", slot_cfg.lr)
        self.weight_decay = exp_dict.get("weight_decay", slot_cfg.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.loss_fn, self.scheduler_type, self.loss_step_hook, self.use_swa = self._resolve_method(slot_cfg)

        self.scheduler = None
        if self.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=getattr(slot_cfg, "min_lr", 1e-6)
            )

        self.swa_model = AveragedModel(self.model) if self.use_swa else None
        self.swa_start_epoch = exp_dict.get("swa_start_epoch", 2)
        self.patience_limit = exp_dict.get("patience_limit", 100)

        self.best_sub_wp = -float("inf")
        self.best_swa_sub_wp = -float("inf")
        self.patience = 0
        self.is_active = True

        self.val_history = []
        self.step_loss_acc = 0.0
        self.step_wp_acc = 0.0
        self.step_count = 0
        self.last_avg_loss = 0.0

    def _resolve_method(self, cfg):
        method = self.method_module
        exp = self.exp_dict
        method_name = getattr(method, "__name__", str(method)).split(".")[-1]

        if method_name == "base_method":
            return base_weighted_pearson_loss, None, None, False
        elif method_name == "cosine_scheduler_method":
            return base_weighted_pearson_loss, "cosine", None, False
        elif method_name == "focal_wp_method":
            gamma = exp.get("focal_gamma", getattr(cfg, "focal_gamma", 2.0))
            loss = lambda p, y: focal_weighted_pearson_loss(p, y, gamma=gamma)
            return loss, "cosine", None, False
        elif method_name == "mse_anchor":
            lambda_init = exp.get("lambda_anchor", getattr(cfg, "lambda_anchor", 0.05))
            hook = lambda s, tot, c: {"lambda_anchor": 0.5 * lambda_init * (1.0 + math.cos(math.pi * min(1.0, s / tot)))}
            return mse_anchor_loss, "cosine", hook, False
        elif method_name == "swa_method":
            return base_weighted_pearson_loss, None, None, True
        elif method_name == "local_global_loss_7_3":
            return local_global_weighted_pearson_loss, None, None, False
        elif method_name == "dynamic_trimmed_method":
            return (lambda p, y: dynamic_trimmed_loss(p, y, keep_ratio=0.70, soft_weight=0.05)), "cosine", None, False
        elif method_name == "asym_wp_loss":
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
    start_time = time.time()
    console = Console()

    progress = Progress(
        UVNameOrSpinnerColumn(name_width=32),
        UVDashedBarColumn(bar_width=32),
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

        model_tasks = {}
        val_tasks = {}

        for s in slots:
            model_tasks[s.name] = progress.add_task(
                s.name,
                total=total_steps,
                is_header=False,
                is_subval=False,
                active=True,
                loss=0.0,
                best_wp=s.best_sub_wp,
                lr=s.lr,
            )
            val_tasks[s.name] = progress.add_task(
                f"{s.name}_subval",
                total=None,
                is_header=False,
                is_subval=True,
                val_line=Text("Waiting for first evaluation...", style="dim italic"),
            )

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
                        model_tasks[s.name],
                        completed=global_step,
                        loss=s.last_avg_loss,
                        best_wp=s.best_sub_wp,
                        lr=s.optimizer.param_groups[0]["lr"],
                        active=s.is_active,
                    )

                # Расчет скорости и оставшегося времени (ETA)
                elapsed = time.time() - start_time
                if elapsed > 0.0:
                    speed = global_step / elapsed
                    rem_steps = max(0, total_steps - global_step)
                    eta_sec = rem_steps / speed if speed > 0 else 0
                    speed_str = f"{speed:.2f} it/s" if speed >= 1.0 else f"{1.0/speed:.2f} s/it"
                    timing_info = f", ~{format_time(eta_sec)} left, {speed_str}"
                else:
                    timing_info = ""

                progress.update(
                    header_task,
                    title=f"Epoch {epoch:02d}/{cfg.max_epochs:02d} | Step {step_in_epoch:04d}/{num_batches:04d} (Global {global_step:05d}{timing_info})",
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

                            prev_wp = s.val_history[-1]["wp"] if s.val_history else None
                            diff = (sub_wp - prev_wp) if prev_wp is not None else 0.0

                            if sub_wp > s.best_sub_wp:
                                s.best_sub_wp = sub_wp
                                s.patience = 0
                                torch.save(s.model.state_dict(), ckpt_path)
                                is_best = True
                            else:
                                s.patience += 1
                                is_best = False

                            s.val_history.append({
                                "step": global_step,
                                "wp": sub_wp,
                                "diff": diff,
                                "is_best": is_best,
                            })

                            current_lr = s.optimizer.param_groups[0]["lr"]
                            logger.log_val_event(epoch, global_step, "sub_10pct", current_lr, res, s.patience)

                            if s.patience >= s.patience_limit:
                                s.is_active = False

                            progress.update(
                                model_tasks[s.name],
                                best_wp=s.best_sub_wp,
                                active=s.is_active,
                            )

                            progress.update(
                                val_tasks[s.name],
                                val_line=format_val_history(
                                    s.val_history,
                                    s.best_sub_wp,
                                    last_res=res,
                                    is_active=s.is_active,
                                    max_items=4,
                                ),
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


generic_train_seed = train_seed