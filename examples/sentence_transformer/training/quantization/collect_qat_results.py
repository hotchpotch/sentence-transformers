#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ResultRow:
    label: str
    path: str
    group: str
    variant: str
    float32: float | None
    int8: float | None
    binary: float | None
    mean3: float | None
    mean_int8_binary: float | None
    train_loss: str | None
    train_batch_size: int | None
    train_binary_mode: str | None
    train_use_int8_range_state: bool | None
    train_precision_warmup_steps: list[int] | None
    train_quantization_warmup_steps: int | None
    quantization_weights: list[float] | None
    eval_quantize_queries: bool | None
    seed: int | None
    num_train_samples: int | None
    num_eval_samples: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect QAT run outputs into markdown/json reports.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        help="Entry format: label|path|group|variant (group/variant optional)",
    )
    parser.add_argument(
        "--entries-file",
        action="append",
        default=[],
        help="Optional file path containing one entry per line. Supports either TSV: label<TAB>path<TAB>group<TAB>variant or entry format label|path|group|variant.",
    )
    return parser.parse_args()


def parse_entry(text: str) -> tuple[str, Path, str, str]:
    parts = text.split("|")
    if len(parts) < 2:
        raise ValueError(f"Invalid --entry format: {text}")
    label = parts[0].strip()
    path = Path(parts[1].strip())
    group = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else "default"
    variant = parts[3].strip() if len(parts) >= 4 and parts[3].strip() else label
    return label, path, group, variant


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_mean(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def build_row(label: str, path: Path, group: str, variant: str) -> ResultRow:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    post = payload.get("post_ndcg10", {})
    float32 = to_float(post.get("float32"))
    int8 = to_float(post.get("int8"))
    binary = to_float(post.get("binary"))
    return ResultRow(
        label=label,
        path=str(path),
        group=group,
        variant=variant,
        float32=float32,
        int8=int8,
        binary=binary,
        mean3=safe_mean([float32, int8, binary]),
        mean_int8_binary=safe_mean([int8, binary]),
        train_loss=config.get("train_loss"),
        train_batch_size=config.get("train_batch_size"),
        train_binary_mode=config.get("train_binary_mode"),
        train_use_int8_range_state=config.get("train_use_int8_range_state"),
        train_precision_warmup_steps=config.get("train_precision_warmup_steps"),
        train_quantization_warmup_steps=config.get("train_quantization_warmup_steps"),
        quantization_weights=config.get("quantization_weights"),
        eval_quantize_queries=config.get("eval_quantize_queries"),
        seed=config.get("seed"),
        num_train_samples=config.get("num_train_samples"),
        num_eval_samples=config.get("num_eval_samples"),
    )


def pick_best(rows: list[ResultRow], metric: str) -> ResultRow | None:
    valid = [row for row in rows if getattr(row, metric) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: getattr(row, metric))


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def main() -> None:
    args = parse_args()

    if not args.entry and not args.entries_file:
        raise ValueError("Provide at least one --entry or --entries-file.")

    rows = []
    all_entries: list[str] = list(args.entry)
    for entries_file in args.entries_file:
        for line in Path(entries_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = line.split("\t")
                label = parts[0].strip()
                path = parts[1].strip() if len(parts) > 1 else ""
                group = parts[2].strip() if len(parts) > 2 else "default"
                variant = parts[3].strip() if len(parts) > 3 else label
                all_entries.append(f"{label}|{path}|{group}|{variant}")
            else:
                all_entries.append(line)

    for entry in all_entries:
        label, path, group, variant = parse_entry(entry)
        rows.append(build_row(label=label, path=path, group=group, variant=variant))

    by_group = {}
    for row in rows:
        by_group.setdefault(row.group, []).append(row)

    best = {
        "float32": asdict(best_float32) if (best_float32 := pick_best(rows, "float32")) else None,
        "int8": asdict(best_int8) if (best_int8 := pick_best(rows, "int8")) else None,
        "binary": asdict(best_binary) if (best_binary := pick_best(rows, "binary")) else None,
        "mean3": asdict(best_mean3) if (best_mean3 := pick_best(rows, "mean3")) else None,
        "mean_int8_binary": (
            asdict(best_mean_i8b) if (best_mean_i8b := pick_best(rows, "mean_int8_binary")) else None
        ),
    }

    md_lines = [
        f"# {args.title}",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for group, group_rows in by_group.items():
        md_lines.extend(
            [
                f"## {group}",
                "",
                "| Label | float32 | int8 | binary | mean3 | mean(int8,binary) | train_loss | batch |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for row in group_rows:
            md_lines.append(
                "| "
                + " | ".join(
                    [
                        row.label,
                        fmt(row.float32),
                        fmt(row.int8),
                        fmt(row.binary),
                        fmt(row.mean3),
                        fmt(row.mean_int8_binary),
                        str(row.train_loss),
                        str(row.train_batch_size),
                    ]
                )
                + " |"
            )
        md_lines.extend(["", "### Details", ""])
        for row in group_rows:
            md_lines.append(f"- `{row.label}` -> `{row.path}`")
            md_lines.append(
                f"  config: train_loss={row.train_loss}, binary_mode={row.train_binary_mode}, "
                f"use_int8_range_state={row.train_use_int8_range_state}, "
                f"precision_warmup={row.train_precision_warmup_steps}, "
                f"quantization_warmup={row.train_quantization_warmup_steps}, "
                f"weights={row.quantization_weights}, eval_quantize_queries={row.eval_quantize_queries}, "
                f"seed={row.seed}, samples={row.num_train_samples}/{row.num_eval_samples}"
            )
        md_lines.append("")

    md_lines.extend(
        [
            "## Best",
            "",
            f"- float32: `{best['float32']['label']}` ({fmt(best['float32']['float32'])})"
            if best["float32"]
            else "- float32: -",
            f"- int8: `{best['int8']['label']}` ({fmt(best['int8']['int8'])})" if best["int8"] else "- int8: -",
            f"- binary: `{best['binary']['label']}` ({fmt(best['binary']['binary'])})"
            if best["binary"]
            else "- binary: -",
            f"- mean3: `{best['mean3']['label']}` ({fmt(best['mean3']['mean3'])})" if best["mean3"] else "- mean3: -",
            f"- mean(int8,binary): `{best['mean_int8_binary']['label']}` ({fmt(best['mean_int8_binary']['mean_int8_binary'])})"
            if best["mean_int8_binary"]
            else "- mean(int8,binary): -",
            "",
        ]
    )

    report_payload = {
        "title": args.title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": [asdict(row) for row in rows],
        "best": best,
    }

    Path(args.output_md).write_text("\n".join(md_lines), encoding="utf-8")
    Path(args.output_json).write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
