# =========================================================
#  jsonl2train.py  (v5)
#  每轮 = 一行，格式: U:xxx\nA:zzz\n   （字面 \n，不是真换行）
#  轮与轮之间: 真换行
#  段与段之间: 空行
#
#  输入:
#    A. {"conversations": [{"from":..., "value":...}, ...]}
#    B. {"prompt": [...], "chosen": "...", "rejected": "..."}
#
#  输出:
#      01.
#      U:xxx\nA:zzz\n
#      U:aaa\nA:bbb\n
#
#      02.
#      ...
#
#  用法：python jsonl2train.py 数据集.jsonl
# =========================================================
import os
import re
import sys
import json
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="输入 JSONL 文件")
    p.add_argument("--out", default="train.txt", help="输出文件（默认 train.txt）")
    return p.parse_args()


def clean_and_flatten(text):
    """清洗：去控制字符；内部换行压成空格，保证每轮只占一行。"""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 内部换行 → 空格（关键：每轮一行）
    text = text.replace("\n", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def role_ok(role):
    role = (role or "").lower()
    if role in ("user", "human", "q"):
        return "U"
    if role in ("assistant", "bot", "gpt", "a"):
        return "A"
    return None


def extract_conversations(item):
    conv = item.get("conversations")
    if not isinstance(conv, list):
        return None
    turns = []
    for msg in conv:
        if not isinstance(msg, dict):
            continue
        r = role_ok(msg.get("from") or msg.get("role"))
        v = msg.get("value") or msg.get("content") or ""
        if r:
            turns.append((r, v))
    return turns or None


def extract_prompt_chosen(item):
    prompt = item.get("prompt")
    chosen = item.get("chosen")
    if not isinstance(prompt, list) or not isinstance(chosen, str):
        return None
    turns = []
    for msg in prompt:
        if not isinstance(msg, dict):
            continue
        r = role_ok(msg.get("role"))
        v = msg.get("content") or msg.get("value") or ""
        if r:
            turns.append((r, v))
    # 末尾补 chosen 作为最后一轮 AI 回答
    if turns and chosen.strip():
        turns.append(("A", chosen))
    return turns or None


def extract_any(item):
    if not isinstance(item, dict):
        return None
    if "conversations" in item:
        return extract_conversations(item)
    if "prompt" in item and "chosen" in item:
        return extract_prompt_chosen(item)
    return None


def turns_to_rounds(turns):
    """
    把 (role, text) 序列按 U/A 配对，返回 [(u_text, a_text), ...]
    - 遇到 U 就暂存
    - 遇到 A 就配成一轮
    - 落单的 U（末尾没有 A）会被丢弃
    """
    rounds = []
    pending_u = None
    for role, text in turns:
        text = clean_and_flatten(text)
        if not text:
            continue
        if role == "U":
            # 如果上一个 U 还没配对，先丢弃它（说明数据有乱）
            pending_u = text
        elif role == "A":
            if pending_u is not None:
                rounds.append((pending_u, text))
                pending_u = None
    return rounds


def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        print(f"❌ 找不到文件: {args.input}")
        return

    kept, dropped, blocks = 0, 0, []

    print(f"[1/2] 读取 {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            turns = extract_any(item)
            if not turns:
                dropped += 1
                continue

            rounds = turns_to_rounds(turns)
            if not rounds:
                dropped += 1
                continue

            # 每轮一行: U:xxx\nA:zzz\n  （\n 是字面两字符）
            lines = []
            for u, a in rounds:
                lines.append(f"U:{u}\\nA:{a}\\n")

            blocks.append(lines)
            kept += 1

    print(f"[2/2] 写入 {args.out} ...")
    with open(args.out, "w", encoding="utf-8") as f:
        for i, lines in enumerate(blocks, 1):
            f.write(f"{i:02d}.\n")
            for ln in lines:
                f.write(ln + "\n")   # 真换行，让每轮占一行
            f.write("\n")            # 段之间空一行

    print(f"\n✅ 完成")
    print(f"   保留: {kept}")
    print(f"   丢弃: {dropped}")
    print(f"   输出: {os.path.abspath(args.out)}")
    print(f"   大小: {os.path.getsize(args.out)} 字节")


if __name__ == "__main__":
    main()