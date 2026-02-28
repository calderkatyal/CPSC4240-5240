#!/usr/bin/env python3
"""
Closest-to-grader benchmark script for HW2.

What it does:
- Compiles your mergesort program (the .cpp you pass in)
- Builds a baseline program that ONLY does std::sort on the same RNG input
- Runs both under:
    - fixed CPU affinity (like grader pinning)
    - fixed thread count (4 by default)
- Reports speedup: baseline_std_sort_time / your_time

Usage:
  python3 bench_like_grader.py /path/to/main.cpp
  python3 bench_like_grader.py /path/to/main.cpp --N 10000000 --threads 4 --cores 16,17,18,19
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASELINE_CPP = r"""
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>  // for omp_get_wtime only

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <N> <seed>\n";
        return 1;
    }
    int N = std::stoi(argv[1]);
    unsigned int seed = std::stoul(argv[2]);

    std::vector<int> data(N);
    std::mt19937 rng(seed);
    for (int i = 0; i < N; ++i) data[i] = (int)rng();

    double t0 = omp_get_wtime();
    std::sort(data.begin(), data.end());
    double t1 = omp_get_wtime();

    std::cout << "BASELINE:" << (t1 - t0) << "\n";
    return 0;
}
"""

def run(cmd, *, env=None):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

def compile_cpp(src_path: Path, out_path: Path):
    cmd = ["g++", "-O3", "-std=c++17", "-fopenmp", "-pthread", str(src_path), "-o", str(out_path)]
    r = run(cmd)
    if r.returncode != 0:
        print("[-] Compile failed:", src_path)
        print(r.stderr)
        sys.exit(1)

def parse_mergesort_time(stdout: str):
    # expects: RESULT:PASS,<time>
    m = re.search(r"RESULT:(PASS|FAIL),([0-9.]+)", stdout)
    if not m:
        return None, None
    return m.group(1), float(m.group(2))

def parse_baseline_time(stdout: str):
    m = re.search(r"BASELINE:([0-9.]+)", stdout)
    if not m:
        return None
    return float(m.group(1))

def affinity_prefix(cores: str):
    # Use taskset if available; matches grader “affinity 16,17,18,19”
    if cores:
        return ["taskset", "-c", cores]
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpp", type=str, help="Path to your HW2 .cpp file")
    ap.add_argument("--N", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cores", type=str, default="0,1,2,3",
                    help="CPU affinity list like '16,17,18,19'. Default simulates 4 logical CPUs.")
    ap.add_argument("--reps", type=int, default=3, help="Run each program this many times and take best (min).")
    args = ap.parse_args()

    cpp_path = Path(args.cpp).resolve()
    if not cpp_path.exists():
        print(f"[-] File not found: {cpp_path}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        yours_exe = td / "yours"
        base_cpp = td / "baseline.cpp"
        base_exe = td / "baseline"

        # Write & compile baseline
        base_cpp.write_text(BASELINE_CPP)
        print(f"[*] Compiling baseline: {base_cpp}")
        compile_cpp(base_cpp, base_exe)

        # Compile your code
        print(f"[*] Compiling yours: {cpp_path}")
        compile_cpp(cpp_path, yours_exe)

        pref = affinity_prefix(args.cores)

        # Environment: force OpenMP to desired threads (your program also sets it via argv)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(args.threads)
        env["OMP_DYNAMIC"] = "FALSE"
        env["OMP_NESTED"] = "TRUE"
        env["OMP_PROC_BIND"] = "TRUE"

        # Run baseline multiple times (take best)
        base_times = []
        for k in range(args.reps):
            cmd = pref + [str(base_exe), str(args.N), str(args.seed)]
            r = run(cmd, env=env)
            t = parse_baseline_time(r.stdout)
            if t is None or r.returncode != 0:
                print("[-] Baseline run failed")
                print(r.stdout)
                print(r.stderr)
                sys.exit(1)
            base_times.append(t)
        base_best = min(base_times)

        # Run yours multiple times (take best)
        your_times = []
        statuses = []
        for k in range(args.reps):
            cmd = pref + [str(yours_exe), str(args.N), str(args.threads), str(args.seed)]
            r = run(cmd, env=env)
            status, t = parse_mergesort_time(r.stdout)
            if status is None or t is None or r.returncode not in (0, 1):
                print("[-] Your program run failed / bad output")
                print(r.stdout)
                print(r.stderr)
                sys.exit(1)
            statuses.append(status)
            your_times.append(t)
        your_best = min(your_times)

        speedup = base_best / your_best if your_best > 0 else float("inf")

        print("\n=== Closest-to-grader benchmark ===")
        print(f"Cores pinned: {args.cores}")
        print(f"Threads:      {args.threads}")
        print(f"N:            {args.N}")
        print(f"Seed:         {args.seed}")
        print(f"Baseline std::sort best of {args.reps}: {base_best:.6f}s")
        print(f"Your code best of {args.reps}:          {your_best:.6f}s")
        print(f"Speedup (sort / yours):                {speedup:.2f}x")
        print(f"Correctness statuses across runs:       {statuses}")

        if speedup >= 3.0 and all(s == "PASS" for s in statuses):
            print("[+] Looks like you meet the >=3x requirement (under these conditions).")
        else:
            print("[-] Under these conditions you do NOT clearly meet >=3x (or you had FAIL runs).")
            print("    Try tuning SERIAL_THRESHOLD / task cutoffs and rerun.")

if __name__ == "__main__":
    main()