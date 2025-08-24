# (필요시) 도구 설치
pip install -U huggingface_hub

# 디렉터리 준비
mkdir -p /workspace/DiffSynth-Studio/models/Qwen/Qwen-Image

# 전체 스냅샷 다운로드 (심볼릭링크 비활성화 권장)
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen-Image",
    local_dir="/workspace/DiffSynth-Studio/models/Qwen/Qwen-Image",
    local_dir_use_symlinks=False
)
PY
