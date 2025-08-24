#!/usr/bin/env python3
import os
import json
import csv
import argparse
import hashlib
from collections import OrderedDict, Counter
from glob import glob

import torch

# 선택적: tqdm 사용 (없어도 동작)
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x

# DiffSynth 파이프라인이 있는 경우 사용
# (Proj 환경에 따라 import 경로가 다를 수 있음)
try:
    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
    HAS_DIFFSYNTH = True
except Exception:
    HAS_DIFFSYNTH = False


def stable_id_from_prompt(p: str) -> str:
    """런타임 및 파이썬 버전과 무관한 안정적 해시 ID 생성"""
    return hashlib.blake2b(p.encode("utf-8"), digest_size=12).hexdigest()


def normalize_prompt(p: str, do_strip: bool, do_norm_ws: bool, do_lower: bool) -> str:
    if p is None:
        return ""
    s = p
    if do_strip:
        s = s.strip()
    if do_norm_ws:
        # 연속 공백을 단일 스페이스로
        s = " ".join(s.split())
    if do_lower:
        s = s.lower()
    return s


def read_prompts_from_csv(csv_path: str, text_column: str | None, sep: str | None):
    """CSV에서 프롬프트 컬럼을 읽어 리스트 반환 (중복 포함)"""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"metadata csv not found: {csv_path}")

    # 컬럼 자동 추론 후보
    candidates = ["caption", "prompt", "text", "prompt_text", "input_text", "captions"]
    prompts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        col = text_column
        if col is None:
            # 자동 추론
            for c in candidates:
                if c in headers:
                    col = c
                    break
        if col is None:
            raise ValueError(
                f"텍스트 컬럼을 찾지 못했습니다. --text-column 로 지정하거나 "
                f"다음 후보 중 하나를 사용하세요: {candidates}. 현재 헤더: {headers}"
            )

        for row in reader:
            raw = row.get(col, "")
            if not raw:
                continue
            if sep and sep in raw:
                items = [x for x in raw.split(sep) if x.strip()]
                prompts.extend(items)
            else:
                prompts.append(raw)
    return prompts


def build_pipe_from_paths(model_paths_csv: str, device: str) -> "QwenImagePipeline":
    if not HAS_DIFFSYNTH:
        raise RuntimeError(
            "DiffSynth 파이프라인 import 실패: "
            "from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig"
        )

    from glob import glob
    uniq_dirs = []
    for part in model_paths_csv.split(","):
        part = part.strip()
        matches = sorted(glob(part)) or [part]
        for m in matches:
            d = os.path.dirname(m) if os.path.isfile(m) else m
            if d not in uniq_dirs:
                uniq_dirs.append(d)

    # 폴더명 기반 힌트 매핑
    def guess_model_name_by_dir(path: str) -> str | None:
        lower = path.lower().rstrip("/")

        # Qwen-Image 표준 폴더명
        if lower.endswith("/transformer"):
            # 감지 실패를 회피하기 위해 변환기를 명시적으로 지정
            return "qwen_image_dit"   # (= QwenImageDiT)
        if lower.endswith("/text_encoder") or lower.endswith("/text-encoder"):
            return "qwen_image_text_encoder"
        if lower.endswith("/vae"):
            return "qwen_image_vae"

        # 그 외 흔한 별칭 대응(선택)
        if lower.endswith("/unet"):
            return "qwen_image_dit"   # 프로젝트에 따라 다를 수 있음
        return None

    model_configs = []
    for d in uniq_dirs:
        hinted = guess_model_name_by_dir(d)
        if hinted is not None:
            # 힌트를 명시해서 아키텍처 자동 감지를 건너뜀
            mc = ModelConfig(path=d, model_name=hinted)
        else:
            mc = ModelConfig(path=d)  # 감지에 맡김(가능한 경우)
        model_configs.append(mc)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs
    )
    return pipe



@torch.inference_mode()
def encode_batch_with_pipe(pipe, texts: list[str], device: str):
    """
    가능한 경우 pipe.encode_prompt 사용,
    아니면 tokenizer/text_encoder 직접 호출.
    반환: (pe, pm) 모두 CPU 텐서
    """
    if hasattr(pipe, "encode_prompt"):
        pe, pm = pipe.encode_prompt(texts)
        # encode_prompt가 반환하는 텐서가 GPU일 수 있으므로 CPU로 이동
        pe = pe.detach().to("cpu")
        pm = pm.detach().to("cpu").to(torch.int64)
        return pe, pm

    # Fallback: tokenizer + text_encoder 직접 접근
    if not hasattr(pipe, "tokenizer") or not hasattr(pipe, "text_encoder"):
        raise RuntimeError(
            "파이프라인에 encode_prompt도, tokenizer/text_encoder도 없어 텍스트 인코딩이 불가합니다."
        )

    tok = pipe.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    tok = {k: v.to(device) for k, v in tok.items()}

    enc = pipe.text_encoder(**tok)
    if hasattr(enc, "last_hidden_state"):
        pe = enc.last_hidden_state
    elif isinstance(enc, (list, tuple)):
        pe = enc[0]
    else:
        pe = enc

    pm = tok["attention_mask"]
    return pe.detach().to("cpu"), pm.detach().to("cpu").to(torch.int64)


def build_prompt_cache(
    model_paths: str,
    prompts: list[str],
    index_path: str,
    device: str = "cpu",
    batch_size: int = 64,
    normalize_ws: bool = False,
    lowercase: bool = False,
    strip: bool = True,
    make_absolute_paths: bool = True,
    resume: bool = True,
):
    """
    prompts(중복 포함)에서 유니크를 추출하여
    {원문 프롬프트: 캐시 파일 절대경로} 형태의 index.json 생성 + 임베딩 저장.
    """
    out_dir = os.path.dirname(index_path)
    emb_dir = os.path.join(out_dir, "embeds")
    os.makedirs(emb_dir, exist_ok=True)

    # 기존 인덱스가 있으면 이어서 (resume)
    index: dict[str, str] = OrderedDict()
    if resume and os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            try:
                index = json.load(f, object_pairs_hook=OrderedDict)
            except Exception:
                print("[WARN] 기존 index.json 파싱 실패. 새로 작성합니다.")
                index = OrderedDict()

    # 전처리(옵션)
    def norm(p: str) -> str:
        return normalize_prompt(p, do_strip=strip, do_norm_ws=normalize_ws, do_lower=lowercase)

    # 빈 문자열 제거
    raw_prompts = [p for p in prompts if isinstance(p, str) and p.strip()]
    # 빈도 카운트(디버그/로그용)
    counts = Counter(raw_prompts)

    # 유니크(원문 기준)
    uniq_raw = list(OrderedDict.fromkeys(raw_prompts))

    # 파이프라인 로드
    pipe = build_pipe_from_paths(model_paths, device=device)

    # 배치 처리
    to_process = []
    for p in uniq_raw:
        key = p  # 키는 **원문** 그대로 사용합니다.
        if key in index and os.path.isfile(index[key]):
            continue
        to_process.append(p)

    print(f"[prompt-cache] 총 프롬프트 수: {len(raw_prompts)} (유니크: {len(uniq_raw)})")
    print(f"[prompt-cache] 새로 인코딩할 개수: {len(to_process)}")
    if len(to_process) == 0:
        print("[prompt-cache] 추가 작업 없음. index.json 갱신만 수행합니다.")

    # 인코딩 & 저장
    for i in tqdm(range(0, len(to_process), batch_size), desc="Encoding prompts"):
        batch = to_process[i:i + batch_size]
        # (선택) 정규화는 저장되는 임베딩에만 사용 (키는 원문)
        batch_norm = [norm(p) for p in batch]

        pe, pm = encode_batch_with_pipe(pipe, batch_norm, device=device)
        # pe: [B, T, C], pm: [B, T]

        for j, raw_p in enumerate(batch):
            pid = stable_id_from_prompt(raw_p)  # 원문 기준으로 ID 생성
            fpath = os.path.join(emb_dir, f"{pid}.pt")
            torch.save(
                {"prompt_emb": pe[j], "prompt_emb_mask": pm[j]},
                fpath
            )
            if make_absolute_paths:
                fpath_to_store = os.path.abspath(fpath)
            else:
                # index.json과 동일 폴더 기준 상대경로가 필요하다면:
                fpath_to_store = os.path.relpath(fpath, start=out_dir)
            index[raw_p] = fpath_to_store

        # 중간 저장(안전)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    # 최종 저장
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"[prompt-cache] 완료: {index_path}")
    # 간단 검증: 임의 샘플 1개
    if len(index) > 0:
        sample_k = next(iter(index.keys()))
        sample_path = index[sample_k]
        ok = os.path.isfile(sample_path)
        print(f"[prompt-cache] 샘플 확인: {sample_k!r} -> {sample_path} (exists={ok})")


def main():
    ap = argparse.ArgumentParser(description="Build prompt cache index (프롬프트 임베딩 캐시 생성기)")
    ap.add_argument("--metadata", required=True, help="데이터셋 메타데이터 CSV 경로 (예: ./pochacco/metadata.csv)")
    ap.add_argument("--text-column", default=None, help="프롬프트가 들어있는 컬럼명 (미지정 시 자동 추론)")
    ap.add_argument("--split-sep", default=None, help="하나의 셀에 복수 프롬프트가 있을 때 구분자 (예: '|||')")
    ap.add_argument("--index-path", required=True, help="생성할 index.json 경로")
    ap.add_argument("--model-paths", required=True, help="콤마로 구분된 모델(글롭) 경로들 (학습 시 --model_id_with_origin_paths와 동일하게 사용 권장)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="인코딩 수행 장치 (기본 cpu)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--normalize-whitespace", action="store_true", help="공백 정규화 수행")
    ap.add_argument("--lowercase", action="store_true", help="소문자화 수행")
    ap.add_argument("--no-strip", action="store_true", help="앞뒤 공백 제거하지 않음")
    ap.add_argument("--relative-paths", action="store_true", help="index.json에 상대경로 저장 (기본은 절대경로)")
    ap.add_argument("--no-resume", action="store_true", help="기존 index.json 무시하고 처음부터 생성")
    args = ap.parse_args()

    prompts = read_prompts_from_csv(args.metadata, args.text_column, args.split_sep)

    build_prompt_cache(
        model_paths=args.model_paths,
        prompts=prompts,
        index_path=args.index_path,
        device=args.device,
        batch_size=args.batch_size,
        normalize_ws=args.normalize_whitespace,
        lowercase=args.lowercase,
        strip=not args.no_strip,
        make_absolute_paths=not args.relative_paths,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
