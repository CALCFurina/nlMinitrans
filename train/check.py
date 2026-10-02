with open("train_split.txt", "rb") as f:
    raw = f.read()

print(f"文件大小: {len(raw)} 字节")
print()

# 整体 UTF-8 校验
try:
    raw.decode("utf-8")
    print("✅ 整体是合法 UTF-8")
except UnicodeDecodeError as e:
    print(f"❌ 整体不是合法 UTF-8")
    print(f"   出错位置: {e.start}-{e.end}")
    print(f"   出错上下文: {raw[max(0,e.start-10):e.end+10]!r}")
    print()

# 逐字节扫描
i, errors = 0, []
while i < len(raw):
    b = raw[i]
    if b < 0x80:
        i += 1
    elif 0xC0 <= b < 0xE0:
        if i+1 >= len(raw) or not (0x80 <= raw[i+1] < 0xC0):
            errors.append(i); i += 1
        else: i += 2
    elif 0xE0 <= b < 0xF0:
        if i+2 >= len(raw) or not (0x80 <= raw[i+1] < 0xC0 and 0x80 <= raw[i+2] < 0xC0):
            errors.append(i); i += 1
        else: i += 3
    else:
        errors.append(i); i += 1

print(f"发现 {len(errors)} 个无效字节位置")
for pos in errors[:20]:
    print(f"  位置 {pos}: 上下文 {raw[max(0,pos-3):pos+6]!r}")
if len(errors) > 20:
    print(f"  ... 还有 {len(errors)-20} 个")

# 顺便看看原始内容
print()
print(f"前 300 字节 (repr):")
print(repr(raw[:300]))