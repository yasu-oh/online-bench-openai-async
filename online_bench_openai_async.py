import argparse
import asyncio
import time
import random
from statistics import mean
from openai import AsyncOpenAI
from typing import List, Tuple

# === ユーティリティ関数 ===
def make_prompt(prompt_tokens: int) -> str:
    base = "You are a helpful assistant. "
    s = base + "Please answer concisely about: "
    topics = [
        "GPU interconnects and NCCL tuning.",
        "EVPN multihoming and LACP behavior.",
        "PagedAttention vs standard attention.",
        "KV cache sizing and throughput trade-offs."
    ]
    while len(s.split()) < prompt_tokens:
        s += random.choice(topics) + " "
    return s.strip()

# === ジョブキュー管理 ===
class JobQueueManager:
    def __init__(self, total_requests: int):
        self.queue = asyncio.Queue()
        for i in range(total_requests):
            self.queue.put_nowait(i)

    async def join(self):
        await self.queue.join()

    def task_done(self):
        self.queue.task_done()

# === ワーカー処理 ===
async def worker(
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    job_manager: JobQueueManager,
    prompt_tokens: int,
    output_tokens: int,
    results: List[Tuple[float, int, int]],
    errors: List[str],
):
    while True:
        try:
            _ = job_manager.queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        prompt = make_prompt(prompt_tokens)
        try:
            async with sem:
                t0 = time.perf_counter()
                rsp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=output_tokens,
                    extra_body={"best_of": 1},
                )
                t1 = time.perf_counter()
            usage = getattr(rsp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) or 0
            out_tok = getattr(usage, "completion_tokens", 0) or 0
            results.append((t1 - t0, in_tok, out_tok))
        except Exception as e:
            errors.append(str(e))
        finally:
            job_manager.task_done()

# === 進捗表示 ===
async def progress_printer(
    total: int,
    results: List[Tuple[float, int, int]],
    errors: List[str],
    start_time: float,
    interval_sec: float = 1.0,
):
    last_line_len = 0
    while True:
        await asyncio.sleep(interval_sec)
        done = len(results) + len(errors)
        now = time.perf_counter()
        elapsed = max(1e-9, now - start_time)
        in_tok = sum(r[1] for r in results)
        out_tok = sum(r[2] for r in results)
        pct = (done / total) * 100 if total > 0 else 100.0
        rps = done / elapsed
        toks_s = out_tok / elapsed
        total_toks_s = (in_tok + out_tok) / elapsed
        remain = max(0, total - done)
        eta = remain / rps if rps > 0 else float("inf")

        line = (
            f"Progress: {done}/{total} ({pct:5.1f}%) | "
            f"RPS(avg): {rps:6.2f} | "
            f"Gen tok/s: {toks_s:7.2f} | "
            f"Total tok/s: {total_toks_s:7.2f} | "
            f"Errors: {len(errors)} | "
            f"ETA: {eta:6.1f}s"
        )
        pad = " " * max(0, last_line_len - len(line))
        print("\r" + line + pad, end="", flush=True)
        last_line_len = len(line)
        if done >= total:
            print()
            return

# === ベンチマーク実行ロジック ===
async def run_benchmark(
    api_base: str,
    api_key: str,
    model: str,
    concurrency: int,
    total_requests: int,
    prompt_tokens: int,
    output_tokens: int,
):
    client = AsyncOpenAI(base_url=api_base, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    results: List[Tuple[float, int, int]] = []
    errors: List[str] = []

    job_manager = JobQueueManager(total_requests)

    # ウォームアップ
    await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Warmup"}],
        temperature=0.0,
        max_tokens=8,
    )

    # ワーカー起動
    tasks = [
        asyncio.create_task(
            worker(
                client, model, sem, job_manager,
                prompt_tokens, output_tokens,
                results, errors
            )
        )
        for _ in range(concurrency)
    ]

    t0 = time.perf_counter()
    prog_task = asyncio.create_task(
        progress_printer(total_requests, results, errors, t0)
    )

    await job_manager.join()
    await asyncio.gather(*tasks, return_exceptions=True)
    await prog_task

    # 結果集計
    t1 = time.perf_counter()
    elapsed = t1 - t0
    lat = [r[0] for r in results]
    in_tok = sum(r[1] for r in results)
    out_tok = sum(r[2] for r in results)
    n = len(results)

    # 出力
    print(f"Requests (ok/err): {n}/{len(errors)}  Concurrency: {concurrency}  Elapsed: {elapsed:.3f}s")
    if elapsed > 0:
        print(f"Req/s: {n/elapsed:.2f}")
        print(f"Generated tokens/s: {out_tok/elapsed:.2f}")
        print(f"Generated tokens/s/user: {out_tok/elapsed/concurrency:.2f}")
        print(f"Total tokens/s (in+out): {(in_tok+out_tok)/elapsed:.2f}")

    if n > 0:
        lat_sorted = sorted(lat)
        p50 = lat_sorted[int(0.50 * (n - 1))] * 1000
        p95 = lat_sorted[int(0.95 * (n - 1))] * 1000
        print(f"p50 latency: {p50:.1f} ms")
        print(f"p95 latency: {p95:.1f} ms")
        print(f"Avg latency: {mean(lat)*1000:.1f} ms")

    if errors:
        print(f"\nFirst 3 errors (of {len(errors)}):")
        for e in errors[:3]:
            print(f"  - {e}")

    print("\n=== Benchmark Configuration ===")
    for k, v in locals().items():
        if k in ["api_base", "api_key", "model", "concurrency", "total_requests", "prompt_tokens", "output_tokens"]:
            print(f"{k:>15}: {v}")

# === メイン関数 ===
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="vllm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--total-requests", type=int, default=1024)
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=1024)
    args = ap.parse_args()

    if args.prompt_tokens <= 0 or args.output_tokens <= 0:
        raise ValueError("prompt_tokens and output_tokens must be positive.")

    await run_benchmark(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        concurrency=args.concurrency,
        total_requests=args.total_requests,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
    )

if __name__ == "__main__":
    asyncio.run(main())
