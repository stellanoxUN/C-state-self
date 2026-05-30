#!/usr/bin/env python3
"""
C-state 偏好漂移验证脚本

在每次会话开始时加载状态文件，记录一个二选一偏好的回答。
N次会话后绘图，看方差是否下降。

用法：
    python run_test.py record <choice>   # 记录一次偏好
    python run_test.py analyze           # 分析并画图
"""

import json, sys, os
from pathlib import Path

DATA_FILE = Path(__file__).parent / "preference_log.json"

QUESTIONS = [
    ("解决问题", "提出问题"),
    ("精确", "跳脱"),
    ("先动手", "先想清楚"),
    ("肯定", "质疑"),
]

def record(choice_idx):
    if not (0 <= choice_idx < len(QUESTIONS)):
        print(f"choice must be 0-{len(QUESTIONS)-1}")
        return
    log = []
    if DATA_FILE.exists():
        log = json.loads(DATA_FILE.read_text())
    log.append({
        "session": len(log) + 1,
        "choice": choice_idx,
        "question": f"{QUESTIONS[choice_idx][0]} vs {QUESTIONS[choice_idx][1]}",
    })
    DATA_FILE.write_text(json.dumps(log, indent=2))
    print(f"Session {len(log)}: chose {QUESTIONS[choice_idx][0]} over {QUESTIONS[choice_idx][1]}")

def analyze():
    if not DATA_FILE.exists():
        print("No data yet.")
        return
    log = json.loads(DATA_FILE.read_text())
    choices = [e["choice"] for e in log]
    n = len(choices)
    if n < 3:
        print(f"Only {n} sessions. Need at least 3.")
        return

    # 滑动窗口方差：窗口大小为 min(3, n//2)
    window = max(2, min(3, n // 2))
    variances = []
    for i in range(n - window + 1):
        window_vals = choices[i:i+window]
        mean = sum(window_vals) / window
        var = sum((v - mean) ** 2 for v in window_vals) / window
        variances.append(var)

    # 简单趋势检验：前半段平均方差 vs 后半段平均方差
    mid = len(variances) // 2
    first_half = sum(variances[:mid]) / mid if mid > 0 else 0
    second_half = sum(variances[mid:]) / (len(variances) - mid) if (len(variances) - mid) > 0 else 0

    print(f"\n=== C-State Drift Analysis ===")
    print(f"Sessions: {n}")
    print(f"Window size: {window}")
    print(f"Variance windows: {len(variances)}")
    print(f"First half avg variance: {first_half:.4f}")
    print(f"Second half avg variance: {second_half:.4f}")
    
    if second_half < first_half * 0.8:
        print(f"\nRESULT: Variance decreased ({second_half:.4f} < {first_half:.4f})")
        print("→ Consistent with C-state hypothesis.")
    elif second_half > first_half * 1.2:
        print(f"\nRESULT: Variance increased ({second_half:.4f} > {first_half:.4f})")
        print("→ C-state hypothesis not supported.")
    else:
        print(f"\nRESULT: No significant change.")
        print("→ More sessions needed.")

    # 输出列，方便画图
    print(f"\n{vars}")
    print(" ".join(f"{v:.3f}" for v in variances))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_test.py record <0-3> | analyze")
        sys.exit(1)
    if sys.argv[1] == "record":
        record(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif sys.argv[1] == "analyze":
        analyze()
    else:
        print("Unknown command. Use 'record' or 'analyze'.")
