import json, re, argparse, os, sys, glob
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple, Optional

# 仅在需要读 parquet 时再导入 pandas，避免无依赖环境报错
def _maybe_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception as e:
        print("ERROR: 读取 Parquet 需要 pandas(和 pyarrow/fastparquet)。请先安装：")
        print("  pip install pandas pyarrow")
        sys.exit(1)

def extract_final_text(harmony_messages: Any) -> str:
    """
    从 Harmony 消息数组中抽取 assistant 的 channel=final 文本，拼接成一个字符串。
    兼容：消息字段为字符串(JSON) / dict / list 等多种情况。
    """
    # harmony_messages 可能是字符串
    if isinstance(harmony_messages, str):
        try:
            harmony_messages = json.loads(harmony_messages)
        except Exception:
            return ""

    finals: List[str] = []
    if not isinstance(harmony_messages, list):
        return ""

    for m in harmony_messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant" and m.get("channel") in ("final", "final_answer", "finalize"):
            parts = m.get("content") or []
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, dict) and p.get("type") == "text":
                        finals.append(p.get("text", ""))
                    elif isinstance(p, str):
                        finals.append(p)
            elif isinstance(parts, str):
                finals.append(parts)

    text = "\n".join([t for t in finals if t is not None])
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def normalize_effort(val: Any) -> str:
    """
    归一化 reasoning effort：返回 'low' / 'middel' / 'high' 三类。
    未知并入 'middel'；容错 middle/mid/med -> middel。
    """
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
        # 其他字符串：尝试数值
        try:
            x = float(v)
            if x <= 0.34: return "low"
            if x >= 0.67: return "high"
            return "middel"
        except Exception:
            return "middel"
    # 数值：分桶
    try:
        x = float(val)
        if x <= 0.34: return "low"
        if x >= 0.67: return "high"
        return "middel"
    except Exception:
        return "middel"

def iter_jsonlike(path: Path) -> Iterable[Dict[str, Any]]:
    """
    支持 .json / .jsonl，且容忍一层包装（data/train/test 等）
    """
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
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
        else:
            raise ValueError("Unsupported top-level JSON structure")

def iter_parquet_files(input_path: Path) -> Iterable[Path]:
    """
    如果输入是目录：遍历 *.parquet；
    如果输入是文件且后缀是 .parquet：直接返回该文件。
    """
    if input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
        if not files:
            # 兼容 HuggingFace 命名 train-00000-of-00003.parquet
            files = sorted(input_path.glob("*.parquet*"))
        for fp in files:
            if fp.is_file() and fp.suffix.startswith(".parquet"):
                yield fp
    elif input_path.is_file() and input_path.suffix.startswith(".parquet"):
        yield input_path

def iter_parquet_records(input_path: Path) -> Iterable[Dict[str, Any]]:
    """
    读取一个或多个 parquet 文件，逐行 yield dict。
    需要 pandas + pyarrow/fastparquet。
    """
    pd = _maybe_import_pandas()
    for pfile in iter_parquet_files(input_path):
        try:
            df = pd.read_parquet(pfile)
        except Exception as e:
            print(f"WARNING: 读取 {pfile} 失败：{e}")
            continue
        # 将 DataFrame 行转成 dict
        for _, row in df.iterrows():
            yield {k: row[k] for k in df.columns}

def guess_effort(rec: Dict[str, Any]) -> Any:
    """
    抽取 reasoning_effort 值，容错多种命名。
    """
    for key in ("reasoning_effort", "reasoning effort", "effort", "reasoningLevel", "reasoning_level"):
        if key in rec:
            return rec.get(key)
    # 尝试嵌套字段（若写在 messages 里）
    msgs = rec.get("gpt-oss-20b-response") or rec.get("gpt_oss_20b_response") or rec.get("response")
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            msgs = None
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                if "reasoning_effort" in m:
                    return m.get("reasoning_effort")
                if "effort" in m:
                    return m.get("effort")
    return None

def get_messages_field(rec: Dict[str, Any]) -> Any:
    return rec.get("gpt-oss-20b-response") or rec.get("gpt_oss_20b_response") or rec.get("response")

def convert_stream(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    out_all: List[Dict[str, Any]] = []
    out_split = {"low": [], "middel": [], "high": []}
    stats = {
        "total_input": 0, "written_all": 0,
        "low": 0, "middel": 0, "high": 0,
        "skipped_no_question": 0, "skipped_no_final": 0
    }

    for rec in records:
        stats["total_input"] += 1
        q = (rec.get("question") or "").strip() if isinstance(rec.get("question"), str) else str(rec.get("question") or "").strip()
        msgs = get_messages_field(rec)
        if not q:
            stats["skipped_no_question"] += 1
            continue

        if msgs is None or (isinstance(msgs, float) and str(msgs) == "nan"):
            stats["skipped_no_final"] += 1
            continue

        ans = extract_final_text(msgs)
        if not ans:
            stats["skipped_no_final"] += 1
            continue

        bucket = normalize_effort(guess_effort(rec))
        item = {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": ans}
            ],
            "meta": {"reasoning_effort": bucket}
        }

        out_all.append(item)
        out_split[bucket].append(item)
        stats["written_all"] += 1
        stats[bucket] += 1

    return out_all, out_split, stats

def save_outputs(out_dir: Path, base_name: str, out_all: List[Dict[str, Any]], out_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    all_path = out_dir / f"{base_name}.json"
    (out_dir / f"{base_name}_low.json").write_text(json.dumps(out_split["low"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{base_name}_middel.json").write_text(json.dumps(out_split["middel"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{base_name}_high.json").write_text(json.dumps(out_split["high"], ensure_ascii=False, indent=2), encoding="utf-8")
    all_path.write_text(json.dumps(out_all, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["output_all"] = str(all_path)
    paths["output_low"] = str(out_dir / f"{base_name}_low.json")
    paths["output_middel"] = str(out_dir / f"{base_name}_middel.json")
    paths["output_high"] = str(out_dir / f"{base_name}_high.json")
    return paths

def main():
    ap = argparse.ArgumentParser(description="Convert gpt-oss (Harmony) to LLaMA-Factory sharegpt, with reasoning_effort splits.")
    ap.add_argument("--input", required=True, help="输入：.json/.jsonl 文件，或 .parquet 文件，或包含 .parquet 分片的目录")
    ap.add_argument("--output-dir", default="./data", help="输出目录（建议指向 LLaMA-Factory 的 data 目录）")
    ap.add_argument("--base-name", default="s1k_gptoss20b_sharegpt", help="输出文件前缀名")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)

    # 决定读取方式
    records_iter: Optional[Iterable[Dict[str, Any]]] = None
    is_parquet = False
    if in_path.is_dir():
        # 目录 => Parquet 分片
        is_parquet = True
    elif in_path.is_file():
        suf = in_path.suffix.lower()
        if suf in (".json", ".jsonl"):
            records_iter = iter_jsonlike(in_path)
        elif suf.startswith(".parquet"):
            is_parquet = True
        else:
            print(f"ERROR: 不支持的输入类型：{in_path}")
            sys.exit(1)
    else:
        print(f"ERROR: 输入路径不存在：{in_path}")
        sys.exit(1)

    if is_parquet:
        records_iter = iter_parquet_records(in_path)

    out_all, out_split, stats = convert_stream(records_iter)
    paths = save_outputs(out_dir, args.base_name, out_all, out_split)

    # 汇总信息
    stats.update(paths)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Done. {stats['written_all']} examples written.")

if __name__ == "__main__":
    main()
