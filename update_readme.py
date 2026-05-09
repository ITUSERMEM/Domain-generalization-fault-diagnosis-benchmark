#!/usr/bin/env python3
"""
Parse training logs and update README.md with results table.
"""

import os
import re
import numpy as np


def parse_log(log_path):
    """Parse a single model's log file."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r', errors='ignore') as f:
        lines = f.readlines()

    # Skip old error logs - find last "Starting"
    start_indices = [i for i, line in enumerate(lines) if 'Starting' in line]
    if len(start_indices) > 1:
        lines = lines[start_indices[-1]:]

    # Extract all evaluation lines
    evals = []
    for line in lines:
        if 'set: Average loss' in line:
            m = re.search(r'(load\d+)_\w+ set:.*Accuracy: (\d+)/(\d+)', line)
            if m:
                name = m.group(1)
                acc = float(m.group(2)) / float(m.group(3)) * 100
                evals.append((name, acc))

    # Target is every 2nd evaluation
    target_evals = [(name, acc) for idx, (name, acc) in enumerate(evals) if idx % 2 == 1]

    # Group by target domain
    task_map = {
        'load3': 'CWRU [0,1,2]→3',
        'load2': 'CWRU [0,1,3]→2',
        'load1': 'CWRU [0,2,3]→1',
        'load0': 'CWRU [1,2,3]→0',
        'load9': 'PU [6,7,8]→9',
        'load8': 'PU [6,7,9]→8',
        'load7': 'PU [6,8,9]→7',
        'load6': 'PU [7,8,9]→6',
    }

    results = {}
    for target_name, task_name in task_map.items():
        accs = [acc for name, acc in target_evals if name == target_name]
        if accs:
            results[task_name] = {
                'max': max(accs),
                'min': min(accs),
                'avg': sum(accs) / len(accs),
                'std': float(np.std(accs)),
                'count': len(accs),
            }

    return results


def update_readme():
    log_dir = 'logs'
    models = ['ERM', 'DANN', 'DDC', 'DCORAL', 'CCDG', 'CNN-C', 'DGNIS', 'IEDGNet']
    tasks = [
        'CWRU [0,1,2]→3', 'CWRU [0,1,3]→2', 'CWRU [0,2,3]→1', 'CWRU [1,2,3]→0',
        'PU [6,7,8]→9', 'PU [6,7,9]→8', 'PU [6,8,9]→7', 'PU [7,8,9]→6',
    ]

    # Collect all results
    all_results = {}
    for model in models:
        log_path = os.path.join(log_dir, f'{model}.log')
        results = parse_log(log_path)
        if results:
            all_results[model] = results

    if not all_results:
        return

    # Build table
    lines = []
    lines.append("## 实验结果\n")
    lines.append("| 模型 | 数据集 | 任务 | 最高 | 最低 | 平均±标准差 | 状态 |")
    lines.append("|------|--------|------|------|------|-------------|------|")

    for model in models:
        if model not in all_results:
            continue
        results = all_results[model]
        for task in tasks:
            if task in results:
                r = results[task]
                status = "✅" if r['count'] >= 450 else "🔄"
                dataset = task.split()[0]
                task_name = task.split()[1]
                lines.append(
                    f"| {model} | {dataset} | {task_name} | {r['max']:.2f}% | {r['min']:.2f}% | {r['avg']:.2f}±{r['std']:.2f}% | {status} |"
                )

    table_str = '\n'.join(lines) + '\n'

    # Update README.md
    readme_path = 'README.md'
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            content = f.read()
    else:
        content = "# Domain Generalization Fault Diagnosis Benchmark\n\n"

    # Replace existing results section or append
    marker = "## 实验结果"
    if marker in content:
        # Find and replace the section
        before = content.split(marker)[0]
        after_parts = content.split(marker)[1].split('\n## ')
        after = ''
        if len(after_parts) > 1:
            after = '\n## ' + '\n## '.join(after_parts[1:])
        content = before + table_str + after
    else:
        content = content.rstrip() + '\n\n' + table_str

    with open(readme_path, 'w') as f:
        f.write(content)

    print(f"README.md updated at {os.path.getmtime(readme_path)}")


if __name__ == '__main__':
    update_readme()
