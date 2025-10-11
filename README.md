# online-bench-openai-async

`online-bench-openai-async` は、**OpenAI互換API（vLLM / TGI / OpenAI API など）** に対して大量の非同期リクエストを送り、**スループット（Req/s・tokens/s）やレイテンシ（p50/p95）** を計測するための軽量ベンチマークツールです。

## 特長

- **非同期I/O対応（AsyncIO）**：数百件のリクエストを効率的に並列実行
- **OpenAI互換API準拠**：`/v1/chat/completions` エンドポイントで動作
- **リアルタイム進捗表示**：RPS・トークン速度・ETA・エラー数を1行で更新
- **詳細な統計出力**：p50/p95/平均レイテンシ・総スループットを計測
- **入出力トークン数を制御可能**：プロンプト長・出力長を柔軟に指定
- **ウォームアップ対応**：初回ロードの影響を除外して安定した計測が可能

## インストール

依存は `openai` のみです。

```bash
git clone https://github.com/<your-username>/online-bench-openai-async.git
cd online-bench-openai-async
pip install openai
```

## 使い方

OpenAI互換エンドポイント（例：vLLMサーバやOpenAI API）に対して実行します。

### vLLM ローカルサーバでの例

```bash
python online_bench_openai_async.py \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key vllm \
  --model qwen2.5-32b-instruct \
  --concurrency 128 \
  --total-requests 1024 \
  --prompt-tokens 1024 \
  --output-tokens 1024
```

### OpenAI公式APIでの例

```bash
python online_bench_openai_async.py \
  --api-base https://api.openai.com/v1 \
  --api-key sk-XXXXXXX \
  --model gpt-4o-mini \
  --concurrency 64 \
  --total-requests 512
```

## コマンドライン引数

| 引数名                | デフォルト                      | 説明                       |
| ------------------ | -------------------------- | ------------------------ |
| `--api-base`       | `http://127.0.0.1:8000/v1` | APIのベースURL               |
| `--api-key`        | `vllm`                     | APIキー（文字列必須、ローカルでも指定が必要） |
| `--model`          | *(必須)*                     | 利用するモデル名                 |
| `--concurrency`    | `128`                      | 同時実行するリクエスト数             |
| `--total-requests` | `1024`                     | 総リクエスト数                  |
| `--prompt-tokens`  | `1024`                     | 入力プロンプトの概算トークン長          |
| `--output-tokens`  | `1024`                     | 出力トークンの最大長               |

## 出力例

```
Progress: 1024/1024 (100.0%) | RPS(avg): 114.32 | Gen tok/s: 92755.13 | Total tok/s: 189334.42 | Errors: 0 | ETA:   0.0s

Requests (ok/err): 1024/0  Concurrency: 128  Elapsed: 8.954s
Req/s: 114.32
Generated tokens/s: 92755.13
Total tokens/s (in+out): 189334.42
p50 latency: 911.3 ms
p95 latency: 1284.2 ms
Avg latency: 893.4 ms

=== Benchmark Configuration ===
     api_base: http://127.0.0.1:8000/v1
      api_key: vllm
        model: qwen2.5-32b-instruct
   concurrency: 128
 total_requests: 1024
 prompt_tokens: 1024
 output_tokens: 1024
```

## 動作概要

1. 総リクエスト数（`--total-requests`）分のジョブをキューに積む
2. 指定された同時実行数（`--concurrency`）のワーカーを非同期で起動
3. 各ワーカーが以下を実行：
   * 疑似プロンプトを生成（`--prompt-tokens`）
   * `chat.completions.create()` を呼び出し
   * 応答時間を計測し、`usage` からトークン統計を記録
4. 全リクエスト完了後に結果を集約し、スループット・レイテンシを出力

## 注意事項

* `extra_body={"best_of": 1}` は **vLLM** 特有の拡張であり、他のAPIでは無視されるかエラーになります。
* 計測しているレイテンシは **API処理時間のみ**（キュー待ちを含まない）。
  キュー待ち時間も含めた「E2Eレイテンシ」を計測したい場合は、`t0 = time.perf_counter()` を `async with sem:` の前に移動してください。
* 一部の実装では `usage` フィールドが返らない場合があり、その際はトークン数が0として集計されます。
* 推奨Pythonバージョンは **3.9以上** です。

## 典型的な利用シナリオ

| シナリオ             | 代表的な設定例                                                   |
| ---------------- | --------------------------------------------------------- |
| vLLM ローカルGPUベンチ  | `--concurrency 256 --total-requests 2048`                 |
| OpenAI API 負荷試験  | `--concurrency 32 --total-requests 512`                   |
| llama.cpp 軽負荷テスト | `--concurrency 8 --prompt-tokens 256 --output-tokens 128` |
