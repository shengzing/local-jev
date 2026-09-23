#!/usr/bin/env python3
"""
Local Jev — 基准测试
对比 Jev 式打分 vs 常规文本生成的延迟和准确率。

用法:
    python benchmark/run_benchmark.py                          # 默认数据集
    python benchmark/run_benchmark.py --dataset support_routing # 指定数据集
    python benchmark/run_benchmark.py --model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.engine import LocalJevEngine


def run_benchmark(
    dataset: list[dict],
    engine: LocalJevEngine,
    choices: dict[str, str],
) -> dict:
    """
    双路径并行基准测试。
    两个 worker 通过 Barrier 同时起跑，各自处理同一批 case。
    """
    results = {"scoring": [], "generation": []}

    starting_line = threading.Barrier(2)

    def scoring_worker():
        starting_line.wait()
        for i, case in enumerate(dataset):
            result = engine.decide(
                state=case["state"],
                question=case["question"],
                choices=choices,
            )
            correct = (result.decision == case["expected"])
            results["scoring"].append({
                "case_id": i,
                "decision": result.decision,
                "expected": case["expected"],
                "correct": correct,
                "latency_ms": result.latency_ms,
                "probabilities": result.probabilities,
            })

    def generation_worker():
        starting_line.wait()
        for i, case in enumerate(dataset):
            result = engine.generate_response(
                state=case["state"],
                question=case["question"],
                choices=list(choices.items()),
                max_tokens=32,
            )
            correct = (result.parsed_choice == case["expected"])
            results["generation"].append({
                "case_id": i,
                "parsed_choice": result.parsed_choice,
                "expected": case["expected"],
                "correct": correct,
                "latency_ms": result.latency_ms,
                "raw_response": result.response,
            })

    t1 = threading.Thread(target=scoring_worker)
    t2 = threading.Thread(target=generation_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # 汇总统计
    summary = {}
    for lane_name, lane_results in results.items():
        latencies = [r["latency_ms"] for r in lane_results]
        correct_count = sum(1 for r in lane_results if r["correct"])
        total = len(lane_results)
        summary[lane_name] = {
            "total_cases": total,
            "correct": correct_count,
            "accuracy": correct_count / total if total > 0 else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "p50_latency_ms": sorted(latencies)[len(latencies)//2] if latencies else 0,
        }

    return {"summary": summary, "details": results}


def main():
    parser = argparse.ArgumentParser(description="Local Jev 基准测试")
    parser.add_argument("--dataset", default="support_routing", help="数据集名称")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    # 加载数据集
    dataset_path = f"datasets/{args.dataset}.jsonl"
    if not os.path.exists(dataset_path):
        print(f"数据集不存在: {dataset_path}")
        sys.exit(1)

    dataset = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    print(f"数据集: {args.dataset} ({len(dataset)} cases)")
    print(f"模型: {args.model}")
    print(f"服务: {args.base_url}\n")

    choices = {
        "A": "billing and payments",
        "B": "technical support",
        "C": "account access",
    }

    engine = LocalJevEngine(base_url=args.base_url, model=args.model)
    result = run_benchmark(dataset, engine, choices)

    # 打印汇总
    print("\n" + "=" * 60)
    print("基准测试结果")
    print("=" * 60)

    for lane, stats in result["summary"].items():
        lane_label = "Jev 式打分" if lane == "scoring" else "常规生成"
        print(f"\n【{lane_label}】")
        print(f"  准确率:   {stats['accuracy']:.2%} ({stats['correct']}/{stats['total_cases']})")
        print(f"  平均延迟: {stats['avg_latency_ms']:.1f} ms")
        print(f"  P50 延迟: {stats['p50_latency_ms']:.1f} ms")
        print(f"  最小延迟: {stats['min_latency_ms']:.1f} ms")
        print(f"  最大延迟: {stats['max_latency_ms']:.1f} ms")

    # 保存详细结果
    out_path = f"benchmark/results/{args.dataset}_{int(time.time())}.json"
    os.makedirs("benchmark/results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
