# =========================================================
#  split_train.py
#  把一个大的 train.txt 切成 N 块小文件
#
#  输入格式（main.py 用的那种）：
#      01.
#      U:xxx\nA:yyy\n
#      <空行>
#      02.
#      U:xxx\nA:yyy\n
#      <空行>
#
#  输出：
#      train_chunks/train_00.txt
#      train_chunks/train_01.txt
#      ...
#
#  用法：
#      python split_train.py train.txt
#      python split_train.py train.txt --num 60 --mb 1.0
# =========================================================
import os
import re
import sys
import argparse

# ---------- 默认参数 ----------
DEFAULT_NUM = 60
DEFAULT_MB  = 1.0
DEFAULT_OUT = "train_chunks"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="输入 train.txt 路径")
    p.add_argument("--num", type=int, default=DEFAULT_NUM,
                   help=f"切几块（默认 {DEFAULT_NUM}）")
    p.add_argument("--mb", type=float, default=DEFAULT_MB,
                   help=f"每块上限 MB（默认 {DEFAULT_MB}）")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"输出目录（默认 {DEFAULT_OUT}）")
    return p.parse_args()


def split_by_marker(content):
    """按 '01.' '02.' 这种标号切分成块"""
    # 找所有 "行首数字加英文句点" 的位置
    pattern = re.compile(r"^[ \t]*\d+\.[ \t]*$", re.MULTILINE)
    matches = list(pattern.finditer(content))

    if not matches:
        # 没有标号，整份当一个块
        return [content]

    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        blocks.append(content[start:end])
    return blocks


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ 找不到文件: {args.input}")
        return

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/3] 读取 {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read()
    total_bytes = len(content.encode("utf-8"))
    print(f"       总字节 {total_bytes/1e6:.2f} MB")

    print(f"[2/3] 切分标号块 ...")
    blocks = split_by_marker(content)
    print(f"       共 {len(blocks)} 个标号块")

    # 每块配额
    chunk_quota = max(1024, int(total_bytes / args.num))
    print(f"       每块配额 ≈ {chunk_quota/1024:.1f} KB")

    # 开始装块
    print(f"[3/3] 开始写入 ...")
    chunk_idx = 0
    chunk_buf = []
    chunk_bytes = 0

    def flush():
        nonlocal chunk_idx, chunk_buf, chunk_bytes
        if not chunk_buf:
            return
        fname = os.path.join(args.out, f"train_{chunk_idx:02d}.txt")
        with open(fname, "w", encoding="utf-8") as out:
            out.write("".join(chunk_buf))
        print(f"       [写入] {fname}  ({chunk_bytes/1024:.1f} KB)")
        chunk_idx += 1
        chunk_buf = []
        chunk_bytes = 0

    for block in blocks:
        b_bytes = len(block.encode("utf-8"))
        # 如果当前块快满了，先落盘（但保证至少一块进得去）
        if chunk_bytes + b_bytes > chunk_quota and chunk_buf:
            flush()
        chunk_buf.append(block)
        chunk_bytes += b_bytes

    # 收尾
    flush()

    print(f"\n✅ 完成")
    print(f"   块数: {chunk_idx}")
    print(f"   目录: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()