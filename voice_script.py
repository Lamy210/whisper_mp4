import os
import torch
import whisper
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from tqdm import tqdm  # 進捗バー用ライブラリのインポート

# .envファイルの読み込み
load_dotenv()

# デバイスの設定（GPUが利用可能な場合はGPUを使用）
device = "cuda" if torch.cuda.is_available() else "cpu"

# Whisperモデルの読み込み
model = whisper.load_model("large-v2", device=device)

# 音声ファイルのパス
audio_path = "input.wav"

# 音声の文字起こし
print("文字起こしを開始します...")
result = model.transcribe(audio_path, language='ja', task='transcribe', verbose=False)
segments = result['segments']

# Hugging Faceのアクセストークンを環境変数から取得
hf_token = os.getenv("HUGGING_FACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGING_FACE_TOKEN環境変数が設定されていません。")

# 話者分離パイプラインの読み込み
print("話者分離パイプラインを読み込み中...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=hf_token
)

# 話者分離の実行
print("話者分離を実行中...")
diarization = diarization_pipeline(audio_path)

# 文字起こし結果と話者情報の統合
def assign_speakers(segments, diarization):
    assigned_segments = []
    print("話者を割り当てています...")
    for segment in tqdm(segments, desc="話者割り当て", unit="segment"):
        start = segment['start']
        end = segment['end']
        text = segment['text']
        speaker_label = 'Unknown'
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if (turn.start <= start <= turn.end) or (turn.start <= end <= turn.end):
                speaker_label = speaker
                break
        assigned_segments.append({
            'start': start,
            'end': end,
            'speaker': speaker_label,
            'text': text
        })
    return assigned_segments

assigned_segments = assign_speakers(segments, diarization)

# 結果の表示とファイルへの保存
output_lines = []
for segment in tqdm(assigned_segments, desc="結果を処理", unit="segment"):
    start = segment['start']
    end = segment['end']
    speaker = segment['speaker']
    text = segment['text']
    line = f"[{start:.2f}-{end:.2f}] 話者 {speaker}: {text}"
    print(line)
    output_lines.append(line)

# テキストファイルに保存
output_file = "output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"\n処理が完了しました。結果は {output_file} に保存されました。")
