# =========================================================
#  jsonl2train_muxue.py
#  专用于 {"system": "...", "conversation": [{"human": "...", "assistant": "..."}, ...]} 格式
#  忽略 system 字段，把多轮 human/assistant 拼成自定义格式
#
#  输出格式：
#      01.
#      U:xxx\nA:zzz\n
#      U:aaa\nA:bbb\n
#      <空行>
#      02.
#      ...
# =========================================================
import os
import re
import json
import sys


def clean_and_flatten(text):
    """清洗：去控制字符，内部换行压成空格，保证每轮只占一行。"""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_rounds(item):
    """从一行 JSON 里提取 (U, A) 轮列表"""
    conv = item.get("conversation")
    if not isinstance(conv, list):
        return None

    rounds = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        u = clean_and_flatten(turn.get("human"))
        a = clean_and_flatten(turn.get("assistant"))
        if u and a:
            rounds.append((u, a))
    return rounds or None


def main():
    if len(sys.argv) < 2:
        print("用法: python jsonl2train_muxue.py 数据集.jsonl [输出文件名]")
        return

    input_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "train_muxue.txt"

    if not os.path.isfile(input_file):
        print(f"❌ 找不到文件: {input_file}")
        return

    kept, dropped, blocks = 0, 0, []

    print(f"[1/2] 读取 {input_file} ...")
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            # ★ 注意：这里直接忽略 item["system"]
            rounds = extract_rounds(item)
            if not rounds:
                dropped += 1
                continue

            # 每轮一行: U:xxx\nA:zzz\n （\n 是字面两字符）
            lines = []
            for u, a in rounds:
                lines.append(f"U:{u}\\nA:{a}\\n")

            blocks.append(lines)
            kept += 1

    print(f"[2/2] 写入 {out_file} ...")
    with open(out_file, "w", encoding="utf-8") as f:
        for i, lines in enumerate(blocks, 1):
            f.write(f"{i:02d}.\n")
            for ln in lines:
                f.write(ln + "\n")   # 真换行，每轮占一行
            f.write("\n")            # 段与段之间空一行

    print(f"\n✅ 完成")
    print(f"   保留: {kept} 条对话组")
    print(f"   丢弃: {dropped} 条无效数据")
    print(f"   输出: {os.path.abspath(out_file)}")
    print(f"   大小: {os.path.getsize(out_file)} 字节")


if __name__ == "__main__":
    main()