# =========================================================
#  Byte-Level Fixed-Role Chatbot (自适应长度 + 多文件 · 验证分批)
# =========================================================
import os
import re
import sys
import glob
import math
import time
import random
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 超参数
# =========================================================
VOCAB   = 256
D       = 40
N_LAYER = 6
N_HEAD  = 5
D_FF    = 80
MAX_SEQ = 1024
D_HEAD  = D // N_HEAD   # 8

# =========================================================
# 模型组件
# =========================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

def apply_rope(x, cos, sin):
    T, dh = x.shape[2], x.shape[-1]
    c = cos[:T].unsqueeze(0).unsqueeze(0)
    s = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.h, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh) + mask
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

class FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.w2(gelu_tanh(self.w1(x)))

class Block(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attention(d, n_heads)
        self.n2 = RMSNorm(d)
        self.ffn = FFN(d, d_ff)
    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.n1(x), cos, sin, mask)
        x = x + self.ffn(self.n2(x))
        return x

class ByteTransformer(nn.Module):
    def __init__(self, V=VOCAB, d=D, n_layer=N_LAYER, n_heads=N_HEAD,
                 d_ff=D_FF, max_seq=MAX_SEQ):
        super().__init__()
        self.tok_emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, d_ff) for _ in range(n_layer)])
        self.nf = RMSNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.cfg = dict(V=V, d=d, n_layer=n_layer, n_heads=n_heads,
                        d_ff=d_ff, max_seq=max_seq)
        self.dh = d // n_heads

        dh = self.dh
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dh, 2).float() / dh))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)
        self.register_buffer("causal_mask",
            torch.triu(torch.full((max_seq, max_seq), float("-inf")), diagonal=1),
            persistent=False)

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        mask = self.causal_mask[:T, :T]
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        x = self.nf(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1), reduction="none")
            m = loss_mask[:, 1:].reshape(-1).float()
            loss = (loss * m).sum() / (m.sum() + 1e-8)
            return logits, loss
        return logits, None

# =========================================================
# 数据解析
# =========================================================
def parse_dialogue_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.compile(r"^[ \t]*\d+\.[ \t]*$", re.MULTILINE)
    matches = list(pattern.finditer(content))
    if not matches:
        groups_src = [content]
    else:
        groups_src = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            groups_src.append(content[start:end])
    groups = []
    for block in groups_src:
        block = block.strip("\r\n")
        if not block:
            continue
        block = block.replace("\\n", "\n")
        block = block.replace("U:", "\x01")
        block = block.replace("A:", "\x02")
        seq = list(block.encode("utf-8"))
        if len(seq) < 3 or 0x02 not in seq:
            continue
        groups.append(seq)
    return groups

def build_loss_mask(seq):
    m = [0.0] * len(seq)
    in_ai = False
    for i, t in enumerate(seq):
        if t == 0x01:
            in_ai = False
        elif t == 0x02:
            in_ai = True
        elif in_ai:
            m[i] = 1.0
            if t == 0x0A:
                in_ai = False
    return m

# =========================================================
# 训练（验证分批）
# =========================================================
def train(data_path, model=None, epochs=60, batch_size=None, lr=1e-3,
          tag=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_num_threads(2)
        if batch_size is None:
            if vram_gb < 5:
                batch_size = 8
            elif vram_gb < 9:
                batch_size = 32
            else:
                batch_size = 64
        accum = max(1, 32 // batch_size)
    else:
        if batch_size is None:
            batch_size = 4
        accum = max(1, 32 // batch_size)
        torch.set_num_threads(max(1, os.cpu_count() or 4))

    use_amp = False

    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}[train] batch={batch_size}  accum={accum}  "
          f"等效={batch_size*accum}  amp={use_amp}  max_seq={MAX_SEQ}")

    groups = parse_dialogue_file(data_path)
    if not groups:
        print(f"{prefix}⚠️ 数据为空，跳过: {data_path}")
        return model

    # ★ 过采样：把小数据复制 N 倍，让它在梯度里"喊得更响"
    OVERSAMPLE = 1
    groups = groups * OVERSAMPLE
    print(f"{prefix}[data] {len(groups)} 组（过采样 ×{OVERSAMPLE}）| "
          f"原始 {len(groups)//OVERSAMPLE} 组")

    lens = [len(g) for g in groups]
    print(f"{prefix}[data] 长度 min={min(lens)} "
          f"max={max(lens)} avg={sum(lens)//len(lens)}")

    N = len(groups)

    def make_batch(bidxs):
        max_len = 1
        for i in bidxs:
            L = min(len(groups[i]), MAX_SEQ)
            if L > max_len:
                max_len = L

        B = len(bidxs)
        x = torch.zeros(B, max_len, dtype=torch.long)
        m = torch.zeros(B, max_len, dtype=torch.float32)
        for j, i in enumerate(bidxs):
            g = groups[i][:max_len]
            n = len(g)
            x[j, :n] = torch.tensor(g, dtype=torch.long)
            m[j, :n] = torch.tensor(build_loss_mask(g), dtype=torch.float32)
        return x.to(device, non_blocking=True), m.to(device, non_blocking=True)

    perm = torch.randperm(N)
    n_val = max(1, N // 5)
    val_idx = perm[:n_val].tolist()
    tr_idx  = perm[n_val:].tolist()
    print(f"{prefix}[data] train={len(tr_idx)}  val={len(val_idx)}")

    if model is None:
        model = ByteTransformer()
        print(f"{prefix}[model] 新建空模型")
    else:
        print(f"{prefix}[model] 从已有权重继续训练")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{prefix}[model] params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=0.01, betas=(0.9, 0.95))

    steps_per_epoch = max(1, len(tr_idx) // batch_size)
    total_opt_steps = max(1, epochs * steps_per_epoch // accum)
    warmup_steps = min(100, max(1, total_opt_steps // 10))
    warmup = torch.optim.lr_scheduler.LinearLR(opt, 0.1, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_opt_steps)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], [warmup_steps])

    best_val, wait = float("inf"), 0
    patience_warn = 10
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        random.shuffle(tr_idx)
        tot, cnt = 0.0, 0
        opt.zero_grad(set_to_none=True)

        for i in range(0, len(tr_idx), batch_size):
            bidx = tr_idx[i:i + batch_size]
            if len(bidx) < 1:
                continue
            x, m = make_batch(bidx)

            _, loss = model(x, targets=x, loss_mask=m)
            (loss / accum).backward()

            tot += loss.item(); cnt += 1

            if (i // batch_size + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

        if (len(tr_idx) // batch_size) % accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()

        # ---- ★ 验证：按 batch_size 分批，不再一次性喂 ----
        model.eval()
        val_sum, val_cnt = 0.0, 0
        with torch.no_grad():
            for vi in range(0, len(val_idx), batch_size):
                vbidx = val_idx[vi:vi + batch_size]
                vx, vm = make_batch(vbidx)
                _, v_loss = model(vx, targets=vx, loss_mask=vm)
                val_sum += v_loss.item() * len(vbidx)
                val_cnt += len(vbidx)
        avg_val = val_sum / max(val_cnt, 1)

        lr_now = opt.param_groups[0]["lr"]
        print(f"{prefix}ep {ep:02d}  train={tot/max(cnt,1):.4f}  "
              f"val={avg_val:.4f}  lr={lr_now:.2e}  "
              f"{time.time()-t0:.1f}s")

        if avg_val < best_val:
            best_val = avg_val; wait = 0
            patience_warn = 10
            torch.save(model.state_dict(), "best.pt")
        else:
            wait += 1
            if wait >= patience_warn:
                print(f"\n{prefix}⚠️ [警告] 验证集 Loss 已连续 {wait} 轮未下降！")
                print(f"{prefix}   train={tot/max(cnt,1):.4f}  "
                      f"val={avg_val:.4f}  best_val={best_val:.4f}")
                try:
                    ans = input(f"{prefix}疑似过拟合，继续？[y/n]: ").strip().lower()
                except EOFError:
                    ans = 'n'

                if ans == 'y':
                    print(f"{prefix}[继续] 再给 1000 轮观察期。")
                    patience_warn += 1000
                else:
                    print(f"{prefix}[停止] 提前结束，加载 best.pt。")
                    break

    model.load_state_dict(torch.load("best.pt", map_location="cpu"))
    return model.cpu()

# =========================================================
# 导出 .btm
# =========================================================
def _quantize_int8(w):
    scale = max(w.abs().max().item() / 127.0, 1e-8)
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale

def _tensor_list(model):
    sd = model.state_dict()
    items = [("emb", sd["tok_emb.weight"])]
    for i in range(N_LAYER):
        items += [
            (f"L{i}.wq", sd[f"blocks.{i}.attn.wq.weight"]),
            (f"L{i}.wk", sd[f"blocks.{i}.attn.wk.weight"]),
            (f"L{i}.wv", sd[f"blocks.{i}.attn.wv.weight"]),
            (f"L{i}.wo", sd[f"blocks.{i}.attn.wo.weight"]),
            (f"L{i}.n1", sd[f"blocks.{i}.n1.weight"]),
            (f"L{i}.w1", sd[f"blocks.{i}.ffn.w1.weight"]),
            (f"L{i}.w2", sd[f"blocks.{i}.ffn.w2.weight"]),
            (f"L{i}.n2", sd[f"blocks.{i}.n2.weight"]),
        ]
    items += [("nf", sd["nf.weight"])]
    return items

def export_btm(model, path, dtype="fp32"):
    entries, blob = [], b""
    off = 0
    for name, w in _tensor_list(model):
        w = w.detach().cpu().float().contiguous()
        ne = w.numel()
        if dtype == "int8":
            q, scale = _quantize_int8(w)
            data = q.numpy().tobytes()
        else:
            scale = 1.0
            data = w.numpy().tobytes()
        entries.append(dict(name=name, offset=off, n_elements=ne,
                            ndim=len(w.shape), shape=list(w.shape), scale=scale))
        off += ne
        blob += data

    with open(path, "wb") as f:
        f.write(b"BTM1")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", 0 if dtype == "fp32" else 1))
        f.write(struct.pack("<I", N_LAYER))
        f.write(struct.pack("<I", D))
        f.write(struct.pack("<I", N_HEAD))
        f.write(struct.pack("<I", D_FF))
        f.write(struct.pack("<I", VOCAB))
        f.write(struct.pack("<I", MAX_SEQ))
        f.write(struct.pack("<I", len(entries)))
        f.write(struct.pack("<I", 1))
        f.write(b"\x00" * 20)
        for e in entries:
            nb = e["name"].encode()
            f.write(nb + b"\x00" * (16 - len(nb)))
            f.write(struct.pack("<I", e["offset"]))
            f.write(struct.pack("<I", e["n_elements"]))
            s = (list(e["shape"]) + [0, 0, 0])[:3]
            f.write(struct.pack("<3H", *s))
            f.write(struct.pack("<B", e["ndim"]))
            f.write(b"\x00")
        for e in entries:
            f.write(struct.pack("<f", e["scale"]))
        cur = 64 + len(entries) * 36
        f.write(b"\x00" * ((64 - cur % 64) % 64))
        f.write(blob)
    print(f"[export] {path}  dtype={dtype}  size={os.path.getsize(path)} B")

def load_btm(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:4] == b"BTM1", "不是 .btm 文件"
    (ver, dtype, n_layer, d, n_heads, d_ff, V, file_max_seq,
     n_tensors, flags) = struct.unpack_from("<10I", raw, 4)
    p = 64
    entries = []
    for _ in range(n_tensors):
        name = raw[p:p + 16].split(b"\x00")[0].decode()
        off, ne = struct.unpack_from("<II", raw, p + 16)
        s0, s1, s2 = struct.unpack_from("<3H", raw, p + 24)
        ndim = raw[p + 30]
        entries.append(dict(name=name, offset=off, n_elements=ne,
                            shape=[s0, s1, s2][:ndim]))
        p += 32
    scales = struct.unpack_from(f"<{n_tensors}f", raw, p)
    p += n_tensors * 4
    p += (64 - p % 64) % 64
    data_bytes = raw[p:]

    model = ByteTransformer(V=V, d=d, n_layer=n_layer, n_heads=n_heads,
                            d_ff=d_ff, max_seq=MAX_SEQ)

    name_map = {"emb": "tok_emb.weight", "nf": "nf.weight"}
    for i in range(n_layer):
        name_map[f"L{i}.wq"] = f"blocks.{i}.attn.wq.weight"
        name_map[f"L{i}.wk"] = f"blocks.{i}.attn.wk.weight"
        name_map[f"L{i}.wv"] = f"blocks.{i}.attn.wv.weight"
        name_map[f"L{i}.wo"] = f"blocks.{i}.attn.wo.weight"
        name_map[f"L{i}.n1"] = f"blocks.{i}.n1.weight"
        name_map[f"L{i}.w1"] = f"blocks.{i}.ffn.w1.weight"
        name_map[f"L{i}.w2"] = f"blocks.{i}.ffn.w2.weight"
        name_map[f"L{i}.n2"] = f"blocks.{i}.n2.weight"

    name_list = [e["name"] for e in entries]
    new_sd = {}
    for k, target in name_map.items():
        e = entries[name_list.index(k)]
        ne, off = e["n_elements"], e["offset"]
        if dtype == 0:
            arr = np.frombuffer(data_bytes, dtype=np.float32, count=ne, offset=off * 4)
            t = torch.from_numpy(arr.copy())
        else:
            arr = np.frombuffer(data_bytes, dtype=np.int8, count=ne, offset=off)
            t = torch.from_numpy(arr.astype(np.float32)) * scales[name_list.index(k)]
        new_sd[target] = t.reshape(e["shape"])
    if "tok_emb.weight" in new_sd:
        new_sd["lm_head.weight"] = new_sd["tok_emb.weight"]
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model

# =========================================================
# PyTorch 推理（3字节重复惩罚）
# =========================================================
@torch.no_grad()
def generate_pytorch(model, prompt_bytes, max_new=200,
                     temperature=0.01, top_k=1, rep_penalty=5.0):
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)

    for _ in range(max_new):
        cond = idx[:, -MAX_SEQ:]
        logits, _ = model(cond)
        logits = logits[:, -1, :] / temperature

        if idx.shape[1] >= 3:
            last_3 = idx[0, -3:].tolist()
            is_repeated = False
            for i in range(idx.shape[1] - 3):
                if idx[0, i:i+3].tolist() == last_3:
                    is_repeated = True
                    break
            if is_repeated:
                for b in last_3:
                    logits[0, b] -= rep_penalty

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)

        if nxt.item() == 0x0A:
            break

    return bytes(idx[0, len(prompt_bytes):].tolist())

# =========================================================
# 手动输入多个文件名
# =========================================================
def pick_data_files(explicit=None):
    raw = ""
    if explicit:
        raw = explicit
    else:
        print("\n" + "=" * 60)
        print("请输入训练文件名（支持多个，用空格或逗号分隔）")
        print("例如: PPPT/train_ascii.txt train_split.txt")
        print("=" * 60)
        try:
            raw = input(">>> ").strip()
        except EOFError:
            sys.exit(1)

    if not raw:
        print("❌ 未输入文件名，退出。")
        sys.exit(0)

    raw = raw.replace(",", " ")
    names = [s for s in raw.split() if s]

    valid = []
    for n in names:
        if os.path.isfile(n):
            valid.append(n)
        else:
            print(f"⚠️ 跳过不存在的文件: {n}")

    if not valid:
        print("❌ 没有任何有效文件，退出。")
        sys.exit(0)

    print(f"\n[选中 {len(valid)} 个文件]")
    for i, f in enumerate(valid):
        print(f"  {i+1}. {f}  ({os.path.getsize(f)} 字节)")

    return valid

# =========================================================
# 主流程
# =========================================================
if __name__ == "__main__":
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    file_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    explicit = " ".join(file_args) if file_args else None

    data_files = pick_data_files(explicit)

    out_dir = os.path.dirname(os.path.abspath(data_files[0])) or "."
    print(f"\n[输出目录] {out_dir}\n")

    fp32_path = os.path.join(out_dir, "model_fp32.btm")
    int8_path = os.path.join(out_dir, "model_int8.btm")

    n_epochs = None
    for i, a in enumerate(sys.argv):
        if a == "--epochs" and i + 1 < len(sys.argv):
            try:
                n_epochs = int(sys.argv[i + 1])
            except ValueError:
                pass
    if n_epochs is None:
        try:
            s = input("每个文件训练轮数 (epochs) [默认 10]: ").strip()
        except EOFError:
            s = ""
        if s == "":
            n_epochs = 10
        elif s.isdigit():
            n_epochs = int(s)
        else:
            n_epochs = 10
    print(f"[config] 每个文件 epochs={n_epochs}\n")

    force_fresh = "--fresh" in flags
    model = None
    if not force_fresh and os.path.isfile(fp32_path):
        try:
            print(f"[load] 发现已有模型 {fp32_path}，继续训练")
            model = load_btm(fp32_path)
            print(f"[load] 成功加载，参数量 "
                  f"{sum(p.numel() for p in model.parameters())}")
        except Exception as e:
            print(f"[load] 加载失败 ({e})，改为新建空模型")
            model = None
    elif force_fresh:
        print("[fresh] --fresh 已指定，从头训练")
    else:
        print(f"[fresh] 未发现 {fp32_path}，新建空模型")

    total = len(data_files)
    for idx, fp in enumerate(data_files):
        tag = f"{idx+1}/{total}"
        print(f"\n{'='*60}")
        print(f" 开始训练 [{tag}] {fp}")
        print(f"{'='*60}")
        model = train(fp, model=model, epochs=n_epochs, lr=1e-3, tag=tag)

        mid_fp32 = os.path.join(out_dir, f"model_after_{idx:02d}.btm")
        export_btm(model, mid_fp32, "fp32")
        print(f"[mid-save] {mid_fp32}")

    print(f"\n{'='*60}")
    print(f" 所有文件训练完成，导出最终模型")
    print(f"{'='*60}")
    export_btm(model, fp32_path, "fp32")
    export_btm(model, int8_path, "int8")

    prompt = b"\x01\xe4\xbd\xa0\xe5\xa5\xbd\x0a\x02"
    out = generate_pytorch(model, prompt, temperature=0.7, top_k=10)
    print("\n[PyTorch 推理]", out.decode("utf-8", errors="replace"))

    m2 = load_btm(int8_path)
    out2 = generate_pytorch(m2, prompt, temperature=0.7, top_k=10)
    print("[INT8 读回]", out2.decode("utf-8", errors="replace"))

    print(f"\n✅ 完成。模型已导出到:")
    print(f"   {fp32_path}")
    print(f"   {int8_path}")