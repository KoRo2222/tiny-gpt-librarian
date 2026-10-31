# webbot

日本語Wikipediaを検索し、見つけた文章を読んで質問に答える小さな言語モデル(RAG)を、トークナイザーからすべて自前で実装するプロジェクトです。LLM自作シリーズの3作目です。

## 概要

- **検索→読む→答える**: ローカルに置いたWikipediaの段落を検索し、取り出した段落から答えを抜き出して回答するモデルを目指します
- **データ**: 文書は [hpprc/jawiki](https://huggingface.co/datasets/hpprc/jawiki)(段落単位の日本語Wikipedia、CC BY-SA 3.0 / GFDL)、学習用のQAは [cl-nagoya/auto-wiki-qa](https://huggingface.co/datasets/cl-nagoya/auto-wiki-qa)(同じWikipediaから生成された質問・答え・根拠段落、CC BY-SA 4.0)、評価は [JSQuAD](https://github.com/yahoojapan/JGLUE)(CC BY-SA 4.0)を使う予定です
- **CPUのみ**で学習できる規模を前提にしています

## 構成

```
src/webbot/tokenizer/bpe.py   バイトレベルBPEトークナイザー(Python版の参照実装と、Rust版の呼び出し)
rust/                         学習とエンコードのRust実装(PyO3 + maturin、モジュール名 webbot_bpe)
tests/test_bpe.py             テスト
```

### トークナイザー

- GPT-2方式のバイトレベルBPEです。事前トークン化は日本語に対応していて、ひらがな・カタカナ・漢字・英字・数字・記号といった文字種の境目で分割します
- 学習には2つの実装があり、`BPETokenizer(backend=...)` で選べます。既定の `"auto"` は、Rust版が入っていればRust版を使います
  - `"python"`: 参照実装。ペア頻度を差分更新し、ヒープは遅延削除方式です
  - `"rust"`: 同じ手順をRustで実装したものです。差分更新は、マージした位置の前後のペアだけを更新します
- 両者は、同点の決め方(記号の文字列の辞書順で大きい方を優先)まで含めて同じマージ列を出します
- エンコードもRust版があり、単語単位のキャッシュを持ちます。事前トークン化(正規表現)はPythonのままです
- 事前トークン化と数え上げは、マルチプロセスで並列化できます
- テストでは、学習結果を「マージごとに全ペアを数え直す素朴な実装」と、Rust版の学習とエンコードの結果を「Python版」と、それぞれ完全一致で比べています

## 現状

トークナイザーまで実装済みです。jawikiの1シャード(6.2万記事、4,800万字)で、Python版とRust版を比べました(i7-13700F)。

| | Python版 | Rust版 |
|---|---|---|
| 学習 1,000万字・語彙8,000 | 54.8秒 | 2.1秒 |
| 学習 4,800万字・語彙16,000 | 212.5秒 | 10.9秒 |
| 学習時のメモリのピーク(4,800万字) | 3.5GB | 1.3GB |
| エンコード | 約12万トークン/秒 | 約300万トークン/秒 |

- 学習の時間には、8プロセスでの事前トークン化も含みます
- 学習したマージ列とエンコード結果は、どちらの条件でもPython版と完全に一致しました
- 語彙16,000のとき、1トークンは平均約1.7文字です

検索器・モデル・学習はこれから実装します。

## セットアップ

Windows・Python 3.12で動作を確認しています。Rust版のビルドには、Rust(MSVCツールチェーン)と Visual Studio Build Tools(C++)が必要です。

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\maturin develop --release -m rust/Cargo.toml
.venv\Scripts\python -m pytest tests
```

Rust版をビルドしなくても、Python版だけで動きます(`backend="python"`)。ただし、Rust版のテストは失敗します。

## License

Copyright (c) 2026 KoRo2. All rights reserved.
