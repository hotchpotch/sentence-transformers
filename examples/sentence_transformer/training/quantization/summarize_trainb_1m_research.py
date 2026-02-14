#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RunRecord:
    round_name: str
    label: str
    tag: str
    result_json: str
    done: bool
    post_float32: float | None
    post_int8: float | None
    post_binary: float | None
    post_mean3: float | None
    post_mean_q: float | None
    best_ckpt_step_mean3: int | None
    best_ckpt_mean3: float | None
    best_ckpt_step_mean_q: int | None
    best_ckpt_mean_q: float | None
    fp32_delta_vs_baseline: float | None
    int8_delta_vs_baseline: float | None
    binary_delta_vs_baseline: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize trainB 1M research results.")
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def to_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def load_rows(tsv_path: Path) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    if not tsv_path.exists():
        return rows
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rows.append((parts[0], parts[1], parts[2], parts[3]))
    # Deduplicate by tag, keeping the latest record.
    by_tag: dict[str, tuple[str, str, str, str]] = {}
    for row in rows:
        by_tag[row[2]] = row
    return list(by_tag.values())


def build_record(
    round_name: str, label: str, tag: str, result_json: str, baseline_ndcg: dict[str, float]
) -> RunRecord:
    path = Path(result_json)
    if not path.exists():
        return RunRecord(
            round_name=round_name,
            label=label,
            tag=tag,
            result_json=result_json,
            done=False,
            post_float32=None,
            post_int8=None,
            post_binary=None,
            post_mean3=None,
            post_mean_q=None,
            best_ckpt_step_mean3=None,
            best_ckpt_mean3=None,
            best_ckpt_step_mean_q=None,
            best_ckpt_mean_q=None,
            fp32_delta_vs_baseline=None,
            int8_delta_vs_baseline=None,
            binary_delta_vs_baseline=None,
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    post = payload.get("post_ndcg10", {})
    f32 = to_float(post.get("float32"))
    i8 = to_float(post.get("int8"))
    binary = to_float(post.get("binary"))

    hist = payload.get("config", {}).get("during_train_eval_history") or []
    best_mean3: tuple[int | None, float | None] = (None, None)
    best_mean_q: tuple[int | None, float | None] = (None, None)
    for entry in hist:
        step = entry.get("step")
        ef = to_float(entry.get("eval_NanoBEIR_mean_float32_float32_cosine_ndcg@10"))
        ei = to_float(entry.get("eval_NanoBEIR_mean_int8_int8_cosine_ndcg@10"))
        eb = to_float(entry.get("eval_NanoBEIR_mean_binary_binary_cosine_ndcg@10"))
        if step is None or ef is None or ei is None or eb is None:
            continue
        m3 = mean([ef, ei, eb])
        mq = mean([ei, eb])
        if m3 is not None and (best_mean3[1] is None or m3 > best_mean3[1]):
            best_mean3 = (int(step), m3)
        if mq is not None and (best_mean_q[1] is None or mq > best_mean_q[1]):
            best_mean_q = (int(step), mq)

    return RunRecord(
        round_name=round_name,
        label=label,
        tag=tag,
        result_json=result_json,
        done=True,
        post_float32=f32,
        post_int8=i8,
        post_binary=binary,
        post_mean3=mean([f32, i8, binary]),
        post_mean_q=mean([i8, binary]),
        best_ckpt_step_mean3=best_mean3[0],
        best_ckpt_mean3=best_mean3[1],
        best_ckpt_step_mean_q=best_mean_q[0],
        best_ckpt_mean_q=best_mean_q[1],
        fp32_delta_vs_baseline=(f32 - baseline_ndcg["float32"]) if f32 is not None else None,
        int8_delta_vs_baseline=(i8 - baseline_ndcg["int8"]) if i8 is not None else None,
        binary_delta_vs_baseline=(binary - baseline_ndcg["binary"]) if binary is not None else None,
    )


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.6f}"


def fmt_step(step: int | None) -> str:
    return "-" if step is None else str(step)


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.tsv))
    baseline_payload = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
    baseline_ndcg = baseline_payload["post_ndcg10"]

    records = [build_record(*row, baseline_ndcg=baseline_ndcg) for row in rows]
    records.sort(key=lambda r: (r.round_name, r.label))

    done = [r for r in records if r.done]
    pending = [r for r in records if not r.done]
    by_round: dict[str, list[RunRecord]] = {}
    for record in records:
        by_round.setdefault(record.round_name, []).append(record)

    md: list[str] = [
        "# trainB 1M Research Report",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Total planned runs: {len(records)}",
        f"- Completed runs: {len(done)}",
        f"- Pending runs: {len(pending)}",
        "",
        "## Baseline Reference",
        "",
        f"- Source: `{args.baseline_json}`",
        f"- float32: {baseline_ndcg['float32']:.6f}",
        f"- int8: {baseline_ndcg['int8']:.6f}",
        f"- binary: {baseline_ndcg['binary']:.6f}",
        "",
    ]

    for round_name, round_records in by_round.items():
        md.extend(
            [
                f"## {round_name}",
                "",
                "| Label | Status | post_f32 | post_i8 | post_bin | post_mean3 | post_mean_q | best_ckpt_mean3(step) | best_ckpt_mean_q(step) | Δf32 vs base | Δi8 vs base | Δbin vs base |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for r in round_records:
            md.append(
                f"| {r.label} | {'done' if r.done else 'pending'} | "
                f"{fmt(r.post_float32)} | {fmt(r.post_int8)} | {fmt(r.post_binary)} | "
                f"{fmt(r.post_mean3)} | {fmt(r.post_mean_q)} | "
                f"{fmt(r.best_ckpt_mean3)} ({fmt_step(r.best_ckpt_step_mean3)}) | "
                f"{fmt(r.best_ckpt_mean_q)} ({fmt_step(r.best_ckpt_step_mean_q)}) | "
                f"{fmt(r.fp32_delta_vs_baseline)} | {fmt(r.int8_delta_vs_baseline)} | {fmt(r.binary_delta_vs_baseline)} |"
            )
        md.append("")

    ranking = sorted(
        [r for r in done if r.post_mean_q is not None],
        key=lambda r: (r.post_mean_q, r.post_mean3 if r.post_mean3 is not None else -1.0),
        reverse=True,
    )
    md.extend(
        [
            "## Ranking (post_mean_q)",
            "",
            "| Rank | Label | Round | post_mean_q | post_mean3 | post_f32 |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for idx, r in enumerate(ranking, start=1):
        md.append(
            f"| {idx} | {r.label} | {r.round_name} | {fmt(r.post_mean_q)} | {fmt(r.post_mean3)} | {fmt(r.post_float32)} |"
        )
    md.append("")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(records),
        "completed_runs": len(done),
        "pending_runs": len(pending),
        "baseline_json": args.baseline_json,
        "baseline_post_ndcg10": baseline_ndcg,
        "records": [asdict(r) for r in records],
    }
    Path(args.output_md).write_text("\n".join(md), encoding="utf-8")
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
