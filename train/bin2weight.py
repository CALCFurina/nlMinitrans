# btm2lua_weights.py
# 读 INT8 .btm，输出紧凑的 Lua 权重文件（字符串 + 长度 + scale）
# 交互式，双击运行
import os
import struct

def main():
    print("=== btm2lua_weights (compact) ===")
    src = input("请输入 .btm 模型文件路径 (默认 model_int8.btm): ").strip()
    if not src:
        src = "model_int8.btm"
    if not os.path.isfile(src):
        print("文件不存在:", src)
        input("按 Enter 退出...")
        return

    dst = input("请输入输出 Lua 文件路径 (默认 weights.lua): ").strip()
    if not dst:
        dst = "weights.lua"

    with open(src, "rb") as f:
        d = f.read()

    def u32(o):
        return d[o] | (d[o+1] << 8) | (d[o+2] << 16) | (d[o+3] << 24)

    assert d[0:4] == b"BTM1", "magic error"
    dtype     = u32(8)
    n_tensors = u32(36)
    print("dtype =", dtype, " n_tensors =", n_tensors)
    if dtype != 1:
        print("仅支持 INT8 模型")
        input("按 Enter 退出...")
        return

    names, offs, nes = [], {}, {}
    for i in range(n_tensors):
        base = 64 + i * 32
        name = d[base:base+16].split(b"\0")[0].decode("ascii")
        names.append(name)
        offs[name] = u32(base + 16)
        nes[name]  = u32(base + 20)

    sbase = 64 + n_tensors * 32
    scale = {}
    for i in range(n_tensors):
        name = names[i]
        scale[name] = struct.unpack("<f", d[sbase + i*4 : sbase + i*4 + 4])[0]

    ds = 64 + n_tensors * 32 + n_tensors * 4
    ds += (64 - ds % 64) % 64

    def key(s):
        return s.replace(".", "_")

    # 把一段字节序列转成 Lua 十进制转义字符串，\ddd 固定三位
    def byte_str(lo, hi):
        out = []
        for i in range(lo, hi):
            out.append("\\%03d" % d[i])
        return "".join(out)

    with open(dst, "w", encoding="utf-8") as f:
        f.write("-- auto-generated from %s\n" % src)
        f.write("-- 权重以字符串存放，运行时用 string.byte 按需读\n\n")
        f.write("local W = {}\n\n")

        # 权重字符串 + 长度
        for name in names:
            off = offs[name]
            ne  = nes[name]
            k   = key(name)
            lo  = ds + off
            hi  = lo + ne
            f.write("W.%s = \"%s\"\n" % (k, byte_str(lo, hi)))
            f.write("W.%s_n = %d\n" % (k, ne))

        f.write("\n")
        # scale
        for name in names:
            k = key(name)
            f.write("W.S_%s = %.9g\n" % (k, scale[name]))

    print("done:", dst)
    input("按 Enter 退出...")

if __name__ == "__main__":
    main()