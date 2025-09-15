import json, re, argparse, sys
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple, Optional

def _maybe_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        print("ERROR: 读取 Parquet 需要 pandas 和 pyarrow。请先安装：")
        print("  pip install pandas pyarrow")
        sys.exit(1)

def _to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return ""

def _flatten_content(parts: Any) -> List[str]:
    out: List[str] = []
    if parts is None:
        return out
    if isinstance(parts, str):
        return [parts]
    if isinstance(parts, list):
        for p in parts:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    out.append(_to_str(p.get("text")))
                elif "text" in p:
                    out.append(_to_str(p.get("text")))
            elif isinstance(p, str):
                out.append(p)
    return out

def extract_analysis_final(messages: Any) -> Tuple[str, str]:
    """提取第一个analysis段和第一个final段的文本"""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return "", ""
    if not isinstance(messages, list):
        return "", ""
    analysis_text, final_text = "", ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        channel = (m.get("channel") or "").lower()
        if channel == "analysis" and not analysis_text:
            analysis_text = "\n".join(_flatten_content(m.get("content"))).strip()
        elif channel in ("final", "final_answer", "finalize") and not final_text:
            final_text = "\n".join(_flatten_content(m.get("content"))).strip()
    return analysis_text, final_text

def normalize_effort(val: Any) -> str:
    if val is None:
        return "middel"
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"low"}:
            return "low"
        if v in {"middel", "middle", "mid", "med"}:
            return "middel"
        if v in {"high", "hi"}:
            return "high"
        try:
            x = float(v)
            if x <= 0.34: return "low"
            if x >= 0.67: return "high"
            return "middel"
        except Exception:
            return "middel"
    try:
        x = float(val)
        if x <= 0.34: return "low"
        if x >= 0.67: return "high"
        return "middel"
    except Exception:
        return "middel"

def iter_jsonlike(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for r in raw:
                yield r
        elif isinstance(raw, dict):
            for k in ("data", "train", "validation", "test", "examples", "items"):
                if k in raw and isinstance(raw[k], list):
                    for r in raw[k]:
                        yield r
                    return
            yield raw

def iter_parquet_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
        if not files:
            files = sorted(input_path.glob("*.parquet*"))
        for fp in files:
            if fp.is_file() and fp.suffix.startswith(".parquet"):
                yield fp
    elif input_path.is_file() and input_path.suffix.startswith(".parquet"):
        yield input_path

def iter_parquet_records(input_path: Path) -> Iterable[Dict[str, Any]]:
    pd = _maybe_import_pandas()
    for pfile in iter_parquet_files(input_path):
        try:
            df = pd.read_parquet(pfile)
        except Exception as e:
            print(f"WARNING: 读取 {pfile} 失败：{e}")
            continue
        for _, row in df.iterrows():
            yield {k: row[k] for k in df.columns}

def guess_effort(rec: Dict[str, Any]) -> Any:
    for key in ("reasoning_effort", "reasoning effort", "effort", "reasoningLevel", "reasoning_level"):
        if key in rec:
            return rec.get(key)
    msgs = rec.get("gpt-oss-20b-response") or rec.get("gpt_oss_20b_response") or rec.get("response")
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            msgs = None
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                if "reasoning_effort" in m: return m.get("reasoning_effort")
                if "effort" in m: return m.get("effort")
    return None

def get_messages_field(rec: Dict[str, Any]) -> Any:
    return rec.get("gpt-oss-20b-response") or rec.get("gpt_oss_20b_response") or rec.get("response")

def convert_stream(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    out_all: List[Dict[str, Any]] = []
    out_split = {"low": [], "middel": [], "high": []}
    stats = {
        "total_input": 0, "written_all": 0,
        "low": 0, "middel": 0, "high": 0,
        "skipped_no_question": 0, "skipped_no_parts": 0
    }
    for rec in records:
        stats["total_input"] += 1
        q = _to_str(rec.get("question")).strip()
        if not q:
            stats["skipped_no_question"] += 1
            continue
        msgs = get_messages_field(rec)
        if msgs is None:
            stats["skipped_no_parts"] += 1
            continue
        analysis, final = extract_analysis_final(msgs)
        if not analysis and not final:
            stats["skipped_no_parts"] += 1
            continue
        combined = (analysis.strip() + "</think>" + final.strip()).strip("</think>")
        bucket = normalize_effort(guess_effort(rec))
        item = {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": combined}
            ],
            "meta": {"reasoning_effort": bucket}
        }
        out_all.append(item)
        out_split[bucket].append(item)
        stats["written_all"] += 1
        stats[bucket] += 1
    return out_all, out_split, stats

def save_outputs(out_dir: Path, base_name: str, out_all: List[Dict[str, Any]], out_split: Dict[str, List[Dict[str, Any]]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{base_name}.json").write_text(json.dumps(out_all, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{base_name}_low.json").write_text(json.dumps(out_split["low"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{base_name}_middel.json").write_text(json.dumps(out_split["middel"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{base_name}_high.json").write_text(json.dumps(out_split["high"], ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Convert GPT-OSS to ShareGPT with analysis+final merged via </think>")
    ap.add_argument("--input", required=True, help="输入文件或目录")
    ap.add_argument("--output-dir", default="./data")
    ap.add_argument("--base-name", default="s1k_gptoss20b_sharegpt_think")
    args = ap.parse_args()
    in_path = Path(args.input)
    if in_path.is_dir():
        records = iter_parquet_records(in_path)
    elif in_path.is_file():
        if in_path.suffix.lower() in (".json", ".jsonl"):
            records = iter_jsonlike(in_path)
        elif in_path.suffix.startswith(".parquet"):
            records = iter_parquet_records(in_path)
        else:
            print(f"不支持的文件类型: {in_path}")
            sys.exit(1)
    else:
        print(f"路径不存在: {in_path}")
        sys.exit(1)
    out_all, out_split, stats = convert_stream(records)
    save_outputs(Path(args.output_dir), args.base_name, out_all, out_split)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
