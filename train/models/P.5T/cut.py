# =========================================================
#  把长对话组拆成短对话组（每组 ≤ max_bytes）
#  用法: python split.py
#        python split.py --max_bytes 1024
#
#  关键：按"轮"切分（U: 为分隔），不切断 UTF-8 字符
# =========================================================
import re
import sys
import argparse


def split_by_rounds(rounds, max_bytes):
    """按字节累加切分，以"轮"为单位（一轮 = U:...A:...）"""
    chunks = []
    current = []
    current_size = 0
    for r in rounds:
        r_bytes = len(r.encode("utf-8"))
        if current and current_size + r_bytes > max_bytes:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(r)
        current_size += r_bytes
    if current:
        chunks.append(current)
    return chunks


def split_dialogue(input_path, output_path,
                   max_bytes=1024, min_rounds_per_group=2):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按标号切分
    pattern = re.compile(r"^[ \t]*\d+\.[ \t]*$", re.MULTILINE)
    matches = list(pattern.finditer(content))
    if not matches:
        print("❌ 没找到标号（如 01. 02.）")
        return

    groups = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        block = content[start:end].strip("\r\n")
        if block:
            groups.append(block)

    print(f"[读入] {len(groups)} 个对话组")

    new_groups = []
    for g in groups:
        # ★ 按 U: 切出每一轮（一轮含 U:...\nA:...\n）
        # 保留原格式（字面 \n 或真换行都行）
        parts = re.split(r"(?=U:)", g)
        rounds = [p for p in parts if p.strip()]
        if not rounds:
            continue

        chunks = split_by_rounds(rounds, max_bytes)

        # 最后一段太短 → 尝试合并到上一段
        if (len(chunks) >= 2
                and len(chunks[-1]) < min_rounds_per_group):
            tail = chunks[-1]
            tail_size = sum(len(r.encode("utf-8")) for r in tail)
            prev_size = sum(len(r.encode("utf-8")) for r in chunks[-2])
            if prev_size + tail_size <= max_bytes:
                chunks[-2].extend(tail)
                chunks.pop()

        new_groups.extend(chunks)

    # 输出（保持原格式）
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(new_groups):
            f.write(f"{i + 1:02d}.\n")
            for r in chunk:
                f.write(r)
                if not r.endswith("\n"):
                    f.write("\n")
            f.write("\n")

    # 统计 + UTF-8 校验
    sizes = []
    bad_utf8 = 0
    for chunk in new_groups:
        text = "".join(chunk)
        try:
            b = text.encode("utf-8")
            b.decode("utf-8")
            sizes.append(len(b))
        except Exception:
            bad_utf8 += 1

    if not sizes:
        print("❌ 没有任何有效对话组")
        return

    print(f"[输出] {len(new_groups)} 个新对话组 → {output_path}")
    print(f"[大小] min={min(sizes)}  max={max(sizes)}  "
          f"avg={sum(sizes) // len(sizes)}")
    over = sum(1 for s in sizes if s > max_bytes)
    if over:
        print(f"⚠️ {over} 组超过 {max_bytes} 字节")
    else:
        print(f"✅ 全部 ≤ {max_bytes} 字节")
    if bad_utf8:
        print(f"❌ {bad_utf8} 组 UTF-8 校验失败")
    else:
        print("✅ UTF-8 校验通过")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="train.txt")
    parser.add_argument("--output", default="train_split.txt")
    parser.add_argument("--max_bytes", type=int, default=1024,
                        help="每组最大字节数（默认 1024）")
    parser.add_argument("--min_rounds", type=int, default=2,
                        help="每组最少轮数（默认 2）")
    args = parser.parse_args()

    split_dialogue(args.input, args.output,
                   max_bytes=args.max_bytes,
                   min_rounds_per_group=args.min_rounds)