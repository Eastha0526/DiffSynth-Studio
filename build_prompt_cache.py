#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, csv, argparse, hashlib
from collections import OrderedDict, Counter
from glob import glob
from typing import List, Tuple, Optional
import torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x

# -----------------------------
# 유틸
# -----------------------------
def stable_id_from_prompt(p: str) -> str:
    return hashlib.blake2b(p.encode("utf-8"), digest_size=12).hexdigest()

def normalize_prompt(p: str, do_strip: bool, do_norm_ws: bool, do_lower: bool) -> str:
    if p is None: return ""
    s = p
    if do_strip: s = s.strip()
    if do_norm_ws: s = " ".join(s.split())
    if do_lower: s = s.lower()
    return s

def read_prompts_from_csv(csv_path: str, text_column: Optional[str], sep: Optional[str]) -> List[str]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[prompt-cache] metadata csv not found: {csv_path}")
    candidates = ["caption","prompt","text","prompt_text","input_text","captions"]
    prompts: List[str] = []
    opened = False
    for enc in ("utf-8","utf-8-sig"):
        try:
            with open(csv_path,"r",encoding=enc,newline="") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                col = text_column
                if col is not None and col not in headers:
                    raise ValueError(f"[prompt-cache] --text-column '{col}' not in headers {headers}")
                if col is None:
                    for c in candidates:
                        if c in headers:
                            col = c; break
                if col is None:
                    raise ValueError(f"[prompt-cache] cannot find text column. headers={headers}, candidates={candidates}")
                for row in reader:
                    raw = row.get(col,"")
                    if not raw: continue
                    if sep and sep in raw:
                        prompts.extend([x for x in raw.split(sep) if x.strip()])
                    else:
                        prompts.append(raw)
                opened = True
            break
        except UnicodeDecodeError:
            continue
    if not opened:
        raise UnicodeDecodeError("utf-8/utf-8-sig", b"", 0, 1, "[prompt-cache] CSV decode failed")
    return prompts

def _list_dirs_from_model_paths(model_paths_csv: str) -> list[str]:
    dirs = []
    for part in (model_paths_csv or "").split(","):
        part = part.strip()
        if not part: continue
        matches = sorted(glob(part)) or [part]
        for m in matches:
            d = os.path.dirname(m) if os.path.isfile(m) else m
            if d not in dirs:
                dirs.append(d)
    return dirs

def find_text_encoder_dir_from_model_paths(model_paths_csv: str) -> Optional[str]:
    for d in _list_dirs_from_model_paths(model_paths_csv):
        name = os.path.basename(d).lower().replace("-","_")
        if name == "text_encoder": return d
    for d in _list_dirs_from_model_paths(model_paths_csv):
        low = os.path.basename(d).lower()
        if "text" in low and "encod" in low:
            return d
    return None

def _likely_tokenizer_dir(path: str) -> bool:
    if not os.path.isdir(path): return False
    files = set(os.listdir(path))
    # 흔한 토크나이저 파일들
    needed_any = {"tokenizer.json","vocab.json","merges.txt","tokenizer.model","spiece.model","tokenizer_config.json"}
    return len(files & needed_any) > 0

def find_tokenizer_dir_from_model_paths(model_paths_csv: str) -> Optional[str]:
    # 우선순위: tokenizer → processor → 기타에 토크나이저 파일이 있는 폴더
    candidates = []
    for d in _list_dirs_from_model_paths(model_paths_csv):
        base = os.path.basename(d).lower().replace("-","_")
        if base == "tokenizer": candidates.append(d)
    for d in _list_dirs_from_model_paths(model_paths_csv):
        base = os.path.basename(d).lower().replace("-","_")
        if base == "processor": candidates.append(d)
    # 그 외 모든 디렉터리에서 토크나이저 파일 보유 여부 검사
    for d in _list_dirs_from_model_paths(model_paths_csv):
        if _likely_tokenizer_dir(d): candidates.append(d)
    for c in candidates:
        if _likely_tokenizer_dir(c):
            return c
    return None

# -----------------------------
# HF 토크나이저/모델 로딩
# -----------------------------
def load_hf_text_encoder(tokenizer_path: str, text_encoder_path: str, device: str):
    from transformers import AutoTokenizer, AutoModel
    if not os.path.isdir(text_encoder_path):
        raise FileNotFoundError(f"[prompt-cache] text_encoder path not found: {text_encoder_path}")
    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(f"[prompt-cache] tokenizer path not found: {tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(text_encoder_path, trust_remote_code=True)

    # pad 토큰 보정
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            try:
                model.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass

    model.eval().to(device)
    return tokenizer, model

@torch.inference_mode()
def encode_batch_hf(tokenizer, model, texts: List[str], device: str, max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    kwargs = dict(return_tensors="pt", padding=True, truncation=True)
    if max_length: kwargs["max_length"] = max_length
    tok = tokenizer(texts, **kwargs)
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model(**tok)
    if hasattr(out, "last_hidden_state"):
        pe = out.last_hidden_state
    elif isinstance(out,(list,tuple)) and len(out)>0:
        pe = out[0]
    elif isinstance(out,dict) and "last_hidden_state" in out:
        pe = out["last_hidden_state"]
    else:
        raise RuntimeError("[prompt-cache] last_hidden_state not found in model output")
    pm = tok["attention_mask"]
    return pe.detach().to("cpu"), pm.detach().to("cpu").to(torch.int64)

# -----------------------------
# 캐시 빌더
# -----------------------------
def build_prompt_cache(
    *,
    prompts: List[str],
    index_path: str,
    text_encoder_path: str,
    tokenizer_path: str,
    device: str = "cpu",
    batch_size: int = 64,
    normalize_ws: bool = False,
    lowercase: bool = False,
    strip: bool = True,
    make_absolute_paths: bool = True,
    resume: bool = True,
    max_length: Optional[int] = None,
):
    out_dir = os.path.dirname(index_path)
    emb_dir = os.path.join(out_dir, "embeds")
    os.makedirs(emb_dir, exist_ok=True)

    index: "OrderedDict[str,str]" = OrderedDict()
    if resume and os.path.isfile(index_path):
        with open(index_path,"r",encoding="utf-8") as f:
            try:
                index = json.load(f, object_pairs_hook=OrderedDict)
            except Exception:
                print("[WARN] 기존 index.json 파싱 실패. 새로 작성합니다."); index = OrderedDict()

    def norm(p: str) -> str:
        return normalize_prompt(p, do_strip=strip, do_norm_ws=normalize_ws, do_lower=lowercase)

    raw_prompts = [p for p in prompts if isinstance(p,str) and p.strip()]
    counts = Counter(raw_prompts)
    uniq_raw = list(OrderedDict.fromkeys(raw_prompts))
    print(f"[prompt-cache] 총 프롬프트 수: {len(raw_prompts)} (유니크: {len(uniq_raw)})")

    tokenizer, model = load_hf_text_encoder(tokenizer_path, text_encoder_path, device=device)

    to_process = []
    for p in uniq_raw:
        if p in index and os.path.isfile(index[p]): continue
        to_process.append(p)
    print(f"[prompt-cache] 새로 인코딩할 개수: {len(to_process)}")
    if len(to_process) == 0:
        with open(index_path,"w",encoding="utf-8") as f: json.dump(index,f,ensure_ascii=False,indent=2)
        print(f"[prompt-cache] 완료: {index_path}")
        return

    for i in tqdm(range(0,len(to_process),batch_size), desc="Encoding prompts"):
        batch = to_process[i:i+batch_size]
        batch_norm = [norm(p) for p in batch]
        pe, pm = encode_batch_hf(tokenizer, model, batch_norm, device=device, max_length=max_length)
        for j, raw_p in enumerate(batch):
            pid = stable_id_from_prompt(raw_p)
            fpath = os.path.join(emb_dir, f"{pid}.pt")
            torch.save({"prompt_emb": pe[j], "prompt_emb_mask": pm[j]}, fpath)
            index[raw_p] = os.path.abspath(fpath) if make_absolute_paths else os.path.relpath(fpath, start=out_dir)
        with open(index_path,"w",encoding="utf-8") as f:
            json.dump(index,f,ensure_ascii=False,indent=2)

    with open(index_path,"w",encoding="utf-8") as f:
        json.dump(index,f,ensure_ascii=False,indent=2)
    print(f"[prompt-cache] 완료: {index_path}")
    if len(index)>0:
        sample_k = next(iter(index.keys()))
        sample_path = index[sample_k]
        print(f"[prompt-cache] 샘플 확인: {sample_k!r} -> {sample_path} (exists={os.path.isfile(sample_path)})")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build prompt cache index (HF text-encoder + tokenizer 분리)")
    ap.add_argument("--metadata", required=True, help="CSV 경로 (예: ./pochacco/metadata.csv)")
    ap.add_argument("--text-column", default=None, help="프롬프트 컬럼명 (미지정 시 자동 탐지)")
    ap.add_argument("--split-sep", default=None, help="한 셀에 여러 프롬프트일 때 구분자")
    ap.add_argument("--index-path", required=True, help="생성할 index.json 경로")

    # 경로 지정: 명시적 또는 model-paths에서 자동탐지
    ap.add_argument("--text-encoder-path", default=None, help="text_encoder 디렉터리 경로(가중치)")
    ap.add_argument("--tokenizer-path", default=None, help="tokenizer/processor 디렉터리 경로(토크나이저 파일들)")
    ap.add_argument("--model-paths", default=None, help="(선택) 콤마구분 경로들에서 text_encoder/tokenizer 자동탐지")

    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--normalize-whitespace", action="store_true")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--no-strip", action="store_true")
    ap.add_argument("--relative-paths", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    te_path = args.text_encoder_path
    tok_path = args.tokenizer_path

    if (te_path is None or tok_path is None) and args.model_paths:
        if te_path is None:
            te_path = find_text_encoder_dir_from_model_paths(args.model_paths)
        if tok_path is None:
            tok_path = find_tokenizer_dir_from_model_paths(args.model_paths)

    if te_path is None:
        raise ValueError("[prompt-cache] text_encoder 경로를 찾지 못했습니다. --text-encoder-path 지정 또는 --model-paths에 포함하세요.")
    if tok_path is None:
        raise ValueError("[prompt-cache] tokenizer 경로를 찾지 못했습니다. --tokenizer-path 지정 또는 --model-paths에 tokenizer/processor 포함하세요.")

    prompts = read_prompts_from_csv(args.metadata, args.text_column, args.split_sep)

    build_prompt_cache(
        prompts=prompts,
        index_path=args.index_path,
        text_encoder_path=te_path,
        tokenizer_path=tok_path,
        device=args.device,
        batch_size=args.batch_size,
        normalize_ws=args.normalize_whitespace,
        lowercase=args.lowercase,
        strip=not args.no_strip,
        make_absolute_paths=not args.relative_paths,
        resume=not args.no_resume,
        max_length=args.max_length,
    )

if __name__ == "__main__":
    main()
