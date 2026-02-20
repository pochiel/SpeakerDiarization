# Speaker Diarization System

音声会議の話者分離 + 文字起こしシステム（Phase 1）

## 概要

| コンポーネント | ライブラリ | 備考 |
|---|---|---|
| 話者分離 | pyannote.audio 3.x | HuggingFaceアカウント必要 |
| 文字起こし | faster-whisper | CPU最適化済み (int8量子化) |

---

## セットアップ

### 1. Hugging Face アカウントとモデル許諾

以下の2つのモデルページにアクセスし、**利用規約に同意**する必要があります：

1. https://huggingface.co/pyannote/speaker-diarization-3.1
2. https://huggingface.co/pyannote/segmentation-3.0

その後、アクセストークンを作成：
https://huggingface.co/settings/tokens

### 2. インストール（Windows）

```bat
install.bat
```

または手動：

```bash
# PyTorch CPU版を先にインストール（必須）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# その他の依存ライブラリ
pip install -r requirements.txt
```

### 3. トークンの設定

`.env.example` を `.env` にコピーしてトークンを設定：

```
HF_TOKEN=hf_xxxxxxxxxxxx
```

---

## 使い方

### 基本

```bash
python transcribe.py meeting.wav
```

出力: `meeting_transcript.json`（同じディレクトリ）

### オプション付き

```bash
# 話者数のヒントを与えると精度向上
python transcribe.py meeting.wav --min-speakers 3 --max-speakers 5

# 話者名マッピング（SPEAKER_00 → 実名）
python transcribe.py meeting.wav --speaker-names "SPEAKER_00=田中,SPEAKER_01=鈴木,SPEAKER_02=佐藤"

# 出力ファイル名を指定
python transcribe.py meeting.wav --output result/2024-01-15.json

# 標準出力への表示を抑制（JSONのみ保存）
python transcribe.py meeting.wav --no-print

# モデルサイズ変更
python transcribe.py meeting.wav --model medium
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--model` | `small` | Whisperモデルサイズ（下記参照） |
| `--language` | `ja` | 言語コード |
| `--min-speakers` | 自動 | 話者数の最小値ヒント |
| `--max-speakers` | 自動 | 話者数の最大値ヒント |
| `--output`, `-o` | 自動 | 出力JSONパス |
| `--speaker-names` | - | 話者名マッピング |
| `--no-print` | - | 標準出力テキストを抑制 |
| `--hf-token` | .env | HFトークン（直接指定） |

---

## 出力フォーマット

```json
{
  "duration": 1800.0,
  "speakers": [
    {
      "speaker_id": "SPEAKER_00",
      "speaker_name": "田中",
      "segments": [
        {
          "start": 0.5,
          "end": 120.3,
          "text": "おはようございます。今日の進捗を報告します..."
        }
      ]
    }
  ]
}
```

標準出力にはタイムスタンプ付きの読みやすい形式も表示されます：

```
================================================================
TRANSCRIPT
================================================================
[0:00:00 - 0:02:00] 田中: おはようございます。今日の進捗を報告します...
[0:02:15 - 0:04:30] 鈴木: ありがとうございます。私の方は...
================================================================
```

---

## モデルサイズの選択

### Whisperモデル（精度 vs 速度）

| モデル | サイズ | 日本語精度 | 30分音声の目安（CPU） |
|---|---|---|---|
| `tiny` | 75 MB | 低 | 約5〜10分 |
| `base` | 142 MB | 中 | 約10〜20分 |
| **`small`** | **466 MB** | **良好 ← 推奨** | **約20〜40分** |
| `medium` | 1.5 GB | 非常に良好 | 約60〜90分 |
| `large-v3` | 3 GB | 最高 | 約120〜180分 |

> CPU性能に強く依存します。実測値は実行時に表示されます。

### pyannoteモデル

話者分離は自動的に `pyannote/speaker-diarization-3.1` を使用します。
30分音声で約10〜30分かかります（CPU依存）。

---

## 初回実行時の動作

初回実行時、以下のモデルが自動ダウンロードされます：

- `~/.cache/huggingface/` に pyannote モデル（約1GB）
- `~/.cache/huggingface/` に Whisperモデル

オフライン環境の場合は事前にモデルファイルを配置し、
`HF_HUB_OFFLINE=1` 環境変数を設定してください。

---

## 対応音声フォーマット

| フォーマット | 対応 |
|---|---|
| WAV | ✅ |
| FLAC | ✅ |
| OGG | ✅ |
| MP3 | ❌（変換が必要） |

MP3を使う場合は事前にWAVに変換してください：

```bash
ffmpeg -i input.mp3 output.wav
```

---

## トラブルシューティング

### `401 Unauthorized` エラー
- HFトークンが正しく設定されているか確認
- モデルの利用規約に同意しているか確認（2つのモデル両方）

### メモリ不足
- モデルサイズを小さくする（`--model tiny` または `--model base`）

### 話者が正しく分離されない
- `--min-speakers` / `--max-speakers` でヒントを与える
- 音声品質を確認（背景ノイズが多い場合は精度低下）

### 文字起こしが空になる
- 音声区間が短すぎる（0.3秒未満は自動スキップ）
- `--language` が正しいか確認

---

## 今後の実装予定

- **Phase 2**: 話者プロファイル管理（embeddingの保存・照合）
- **Phase 3**: LLM統合（llama-cpp-python による要約・タスク抽出）
