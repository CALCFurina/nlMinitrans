/* =========================================================
 *  ai.c  —  字节级对话机器人  C 推理端（多轮上下文版）
 *
 *  模型: N=6, d=40, h=5, d_head=8, d_ff=80, seq=1024
 *  权重: INT8 或 FP32 (.btm)
 *  KV:   FP16 (软件模拟)
 *
 *  特性：
 *   - 程序内输入模型路径
 *   - UTF-8 控制台 I/O
 *   - 温度=0.01，top_k=1（极度保守）
 *   - 3字节（单汉字）重复惩罚
 *   - ★ 多轮上下文：保留完整对话历史
 *
 *  编译: gcc -O2 -o ai.exe ai.c -lm
 * ========================================================= */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif

/* ---------- 编译期常量 ---------- */
#define MAX_TENSORS  128
#define MAX_LAYERS   32
#define D_MAX        64
#define FF_MAX       256
#define SEQ_MAX      2048
#define VOCAB        256
#define HIST_MAX     4096     /* 历史缓冲区最大值（字节） */

/* ---------- 文件格式 ---------- */
#pragma pack(push, 1)
typedef struct {
    char     magic[4];
    uint32_t version;
    uint32_t dtype;
    uint32_t n_layers;
    uint32_t d_model;
    uint32_t n_heads;
    uint32_t d_ff;
    uint32_t vocab_size;
    uint32_t max_seq;
    uint32_t n_tensors;
    uint32_t flags;
    uint32_t reserved[5];
} Header;

typedef struct {
    char     name[16];
    uint32_t offset;
    uint32_t n_elements;
    uint16_t shape[3];
    uint8_t  ndim;
    uint8_t  reserved;
} TensorEntry;
#pragma pack(pop)

typedef struct {
    Header       hdr;
    TensorEntry  tensors[MAX_TENSORS];
    float       *data;
    uint32_t     total_elems;
    int          id_emb, id_nf;
    int          id_wq[MAX_LAYERS], id_wk[MAX_LAYERS], id_wv[MAX_LAYERS],
                 id_wo[MAX_LAYERS], id_w1[MAX_LAYERS], id_w2[MAX_LAYERS],
                 id_n1[MAX_LAYERS], id_n2[MAX_LAYERS];
} Model;

/* =========================================================
 * FP16 软转换
 * ========================================================= */
static inline uint16_t f32_to_f16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3FF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = ((u >> 13) | 0x400) >> (1 - exp);
        return (uint16_t)(sign | mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    uint32_t round = (u >> 12) & 1;
    return (uint16_t)(sign | (exp << 10) | (mant + round));
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) u = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            u = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000 | (mant << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &u, 4);
    return f;
}

/* =========================================================
 * RoPE 预计算
 * ========================================================= */
static float *g_rope_cos = NULL, *g_rope_sin = NULL;
static int    g_rope_half = 0;

static void rope_init(int d_head, int max_seq, float base) {
    int half = d_head / 2;
    g_rope_half = half;
    g_rope_cos = malloc(sizeof(float) * max_seq * half);
    g_rope_sin = malloc(sizeof(float) * max_seq * half);
    for (int p = 0; p < max_seq; p++)
        for (int i = 0; i < half; i++) {
            float theta = powf(base, -2.0f * i / d_head);
            float a = p * theta;
            g_rope_cos[p * half + i] = cosf(a);
            g_rope_sin[p * half + i] = sinf(a);
        }
}

static inline void rope_apply(float *vec, int pos, int d_head) {
    int half = d_head / 2;
    const float *c = g_rope_cos + (size_t)pos * half;
    const float *s = g_rope_sin + (size_t)pos * half;
    for (int i = 0; i < half; i++) {
        float x1 = vec[i], x2 = vec[i + half];
        vec[i]        = x1 * c[i] - x2 * s[i];
        vec[i + half] = x1 * s[i] + x2 * c[i];
    }
}

/* =========================================================
 * 加载 .btm
 * ========================================================= */
static int find_tensor(const Model *m, const char *name) {
    for (uint32_t i = 0; i < m->hdr.n_tensors; i++)
        if (strcmp(m->tensors[i].name, name) == 0) return (int)i;
    return -1;
}

int model_load(const char *path, Model *m) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return -1; }

    if (fread(&m->hdr, sizeof(Header), 1, f) != 1) { fclose(f); return -2; }
    if (memcmp(m->hdr.magic, "BTM1", 4) != 0)      { fclose(f); return -3; }

    uint32_t nt = m->hdr.n_tensors;
    if (nt > MAX_TENSORS) { fclose(f); return -4; }
    if (fread(m->tensors, sizeof(TensorEntry), nt, f) != nt) { fclose(f); return -5; }

    float *scales = malloc(sizeof(float) * nt);
    if (fread(scales, sizeof(float), nt, f) != nt) {
        free(scales); fclose(f); return -6;
    }

    uint32_t total = 0;
    for (uint32_t i = 0; i < nt; i++) {
        uint32_t e = m->tensors[i].offset + m->tensors[i].n_elements;
        if (e > total) total = e;
    }
    m->total_elems = total;

    m->data = malloc(sizeof(float) * total);
    if (!m->data) { free(scales); fclose(f); return -7; }

    long base = sizeof(Header) + (long)nt * sizeof(TensorEntry)
                                + (long)nt * sizeof(float);
    base += (64 - base % 64) % 64;
    fseek(f, base, SEEK_SET);

    if (m->hdr.dtype == 0) {
        size_t need = sizeof(float) * total;
        if (fread(m->data, 1, need, f) != need) {
            free(scales); free(m->data); fclose(f); return -8;
        }
    } else {
        uint8_t *raw = malloc(total);
        if (!raw || fread(raw, 1, total, f) != total) {
            free(raw); free(scales); free(m->data); fclose(f); return -9;
        }
        for (uint32_t i = 0; i < nt; i++) {
            const int8_t *src = (const int8_t *)raw + m->tensors[i].offset;
            float *dst = m->data + m->tensors[i].offset;
            float s = scales[i];
            uint32_t ne = m->tensors[i].n_elements;
            for (uint32_t j = 0; j < ne; j++) dst[j] = (float)src[j] * s;
        }
        free(raw);
    }
    free(scales);
    fclose(f);

    m->id_emb = find_tensor(m, "emb");
    m->id_nf  = find_tensor(m, "nf");
    char buf[32];
    for (uint32_t L = 0; L < m->hdr.n_layers; L++) {
        #define LOOK(field, tag) do { \
            snprintf(buf, sizeof(buf), "L%u." tag, L); \
            m->id_##field[L] = find_tensor(m, buf); \
        } while (0)
        LOOK(wq, "wq"); LOOK(wk, "wk"); LOOK(wv, "wv"); LOOK(wo, "wo");
        LOOK(n1, "n1"); LOOK(w1, "w1"); LOOK(w2, "w2"); LOOK(n2, "n2");
        #undef LOOK
    }
    return 0;
}

/* =========================================================
 * 基础算子
 * ========================================================= */
static inline const float *T(const Model *m, int id) {
    return m->data + m->tensors[id].offset;
}

static void rmsnorm(float *x, const float *g, int d) {
    float ss = 0;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / d + 1e-6f);
    for (int i = 0; i < d; i++) x[i] = x[i] * inv * g[i];
}

static void matvec(const float *W, const float *x, float *y, int o_dim, int i_dim) {
    for (int o = 0; o < o_dim; o++) {
        const float *row = W + (size_t)o * i_dim;
        float acc = 0;
        for (int i = 0; i < i_dim; i++) acc += row[i] * x[i];
        y[o] = acc;
    }
}

static inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f *
           (x + 0.044715f * x * x * x)));
}

typedef struct {
    uint16_t *k;
    uint16_t *v;
    int       len;
} KVCache;

/* =========================================================
 * 单步前向
 * ========================================================= */
static void forward_one(Model *m, int token, int pos,
                        KVCache *kv, float *logits) {
    const int d  = m->hdr.d_model;
    const int h  = m->hdr.n_heads;
    const int dh = d / h;
    const int ff = m->hdr.d_ff;
    const int nl = m->hdr.n_layers;
    const int V  = m->hdr.vocab_size;
    const int ms = m->hdr.max_seq;

    static float x[D_MAX], xn[D_MAX], q[D_MAX], k[D_MAX], v[D_MAX],
                 ao[D_MAX], proj[D_MAX], ff1[FF_MAX], ff2[D_MAX];
    static float scores[SEQ_MAX];

    memcpy(x, T(m, m->id_emb) + (size_t)token * d, sizeof(float) * d);

    for (int L = 0; L < nl; L++) {
        memcpy(xn, x, sizeof(float) * d);
        rmsnorm(xn, T(m, m->id_n1[L]), d);

        matvec(T(m, m->id_wq[L]), xn, q, d, d);
        matvec(T(m, m->id_wk[L]), xn, k, d, d);
        matvec(T(m, m->id_wv[L]), xn, v, d, d);

        for (int hh = 0; hh < h; hh++) {
            rope_apply(q + hh * dh, pos, dh);
            rope_apply(k + hh * dh, pos, dh);
        }

        uint16_t *kL = kv->k + (size_t)L * ms * d;
        uint16_t *vL = kv->v + (size_t)L * ms * d;
        uint16_t *kW = kL + (size_t)pos * d;
        uint16_t *vW = vL + (size_t)pos * d;
        for (int i = 0; i < d; i++) {
            kW[i] = f32_to_f16(k[i]);
            vW[i] = f32_to_f16(v[i]);
        }

        const float inv_sqrt_dh = 1.0f / sqrtf((float)dh);
        for (int hh = 0; hh < h; hh++) {
            const float *qh = q + hh * dh;
            float *oh = ao + hh * dh;
            float maxs = -1e30f;
            for (int t = 0; t <= pos; t++) {
                const uint16_t *kh16 = kL + (size_t)t * d + hh * dh;
                float s = 0;
                for (int i = 0; i < dh; i++)
                    s += qh[i] * f16_to_f32(kh16[i]);
                s *= inv_sqrt_dh;
                scores[t] = s;
                if (s > maxs) maxs = s;
            }
            float sum = 0;
            for (int t = 0; t <= pos; t++) {
                float e = expf(scores[t] - maxs);
                scores[t] = e;
                sum += e;
            }
            float inv_sum = 1.0f / sum;
            for (int i = 0; i < dh; i++) oh[i] = 0;
            for (int t = 0; t <= pos; t++) {
                float w = scores[t] * inv_sum;
                const uint16_t *vh16 = vL + (size_t)t * d + hh * dh;
                for (int i = 0; i < dh; i++)
                    oh[i] += w * f16_to_f32(vh16[i]);
            }
        }

        matvec(T(m, m->id_wo[L]), ao, proj, d, d);
        for (int i = 0; i < d; i++) x[i] += proj[i];

        memcpy(xn, x, sizeof(float) * d);
        rmsnorm(xn, T(m, m->id_n2[L]), d);
        matvec(T(m, m->id_w1[L]), xn, ff1, ff, d);
        for (int i = 0; i < ff; i++) ff1[i] = gelu_tanh(ff1[i]);
        matvec(T(m, m->id_w2[L]), ff1, ff2, d, ff);
        for (int i = 0; i < d; i++) x[i] += ff2[i];
    }

    rmsnorm(x, T(m, m->id_nf), d);
    const float *emb = T(m, m->id_emb);
    for (int vv = 0; vv < V; vv++) {
        const float *row = emb + (size_t)vv * d;
        float s = 0;
        for (int i = 0; i < d; i++) s += row[i] * x[i];
        logits[vv] = s;
    }
}

/* =========================================================
 * 采样（带 3 字节重复惩罚）
 * ========================================================= */
static int sample_with_penalty(const float *logits, int n, float temp, int top_k,
                              unsigned *seed, uint8_t *history, int hist_len,
                              float rep_penalty) {
    static float buf[VOCAB];
    static int   idx[VOCAB];

    float adjusted[VOCAB];
    for (int i = 0; i < n; i++) adjusted[i] = logits[i];

    if (hist_len >= 3) {
        uint8_t b0 = history[hist_len - 3];
        uint8_t b1 = history[hist_len - 2];
        uint8_t b2 = history[hist_len - 1];
        int repeated = 0;
        for (int i = 0; i <= hist_len - 3; i++) {
            if (history[i] == b0 && history[i+1] == b1 && history[i+2] == b2) {
                repeated = 1; break;
            }
        }
        if (repeated) {
            adjusted[b0] -= rep_penalty;
            adjusted[b1] -= rep_penalty;
            adjusted[b2] -= rep_penalty;
        }
    }

    float maxv = adjusted[0];
    for (int i = 1; i < n; i++) if (adjusted[i] > maxv) maxv = adjusted[i];

    for (int i = 0; i < n; i++) {
        buf[i] = expf((adjusted[i] - maxv) / temp);
        idx[i] = i;
    }

    if (top_k > 0 && top_k < n) {
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i + 1; j < n; j++)
                if (buf[j] > buf[best]) best = j;
            if (best != i) {
                float tf = buf[i]; buf[i] = buf[best]; buf[best] = tf;
                int   ti = idx[i]; idx[i] = idx[best]; idx[best] = ti;
            }
        }
        n = top_k;
    }

    float sum = 0;
    for (int i = 0; i < n; i++) sum += buf[i];

    *seed = *seed * 1103515245u + 12345u;
    float r = ((*seed >> 8) & 0xFFFFFF) / (float)0x1000000 * sum;

    float c = 0;
    for (int i = 0; i < n; i++) {
        c += buf[i];
        if (c >= r) return idx[i];
    }
    return idx[n - 1];
}

/* =========================================================
 * 生成（带 3 字节重复惩罚）
 * ========================================================= */
static void generate(Model *m, const uint8_t *prompt, int plen,
                     int max_new, float temp, int top_k,
                     uint8_t *out, int *out_len) {
    int ms = m->hdr.max_seq;
    KVCache kv;
    kv.k = calloc((size_t)m->hdr.n_layers * ms * m->hdr.d_model, sizeof(uint16_t));
    kv.v = calloc((size_t)m->hdr.n_layers * ms * m->hdr.d_model, sizeof(uint16_t));
    kv.len = 0;

    float *logits = malloc(sizeof(float) * m->hdr.vocab_size);
    int pos = 0;

    for (int i = 0; i < plen && pos < ms; i++) {
        forward_one(m, prompt[i], pos, &kv, logits);
        pos++;
    }

    unsigned seed = 0x12345678u ^ (unsigned)time(NULL);
    *out_len = 0;
    uint8_t history[2048];
    int hist_len = 0;

    for (int s = 0; s < max_new && pos < ms; s++) {
        int nxt = sample_with_penalty(logits, m->hdr.vocab_size, temp, top_k,
                                      &seed, history, hist_len, 5.0f);
        out[(*out_len)++] = (uint8_t)nxt;
        history[hist_len++] = (uint8_t)nxt;

        if (nxt == 0x0A) break;

        forward_one(m, nxt, pos, &kv, logits);
        pos++;
    }

    free(logits);
    free(kv.k);
    free(kv.v);
}

/* =========================================================
 * UTF-8 I/O 初始化
 * ========================================================= */
static void utf8_init(void) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
#endif
}

/* =========================================================
 * ★ 多轮上下文 REPL
 * ========================================================= */
static void repl(Model *m) {
    char line[1024];
    uint8_t out[1024];

    /* 历史缓冲区：保存所有轮次 "01...0A 02...0A" */
    static uint8_t history[HIST_MAX];
    int hist_len = 0;
    int max_hist = m->hdr.max_seq - 128;  /* 给新输入留余量 */
    if (max_hist > HIST_MAX) max_hist = HIST_MAX;

    printf("\n进入多轮上下文模式。\n");
    printf("  q / quit  = 退出\n");
    printf("  clear     = 清空对话历史\n");
    printf("  其他      = 与 AI 对话（会自动携带历史）\n");

    while (1) {
        printf("\n>>> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        int n = (int)strlen(line);
        while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
        if (n == 0) continue;
        if (strcmp(line, "q") == 0 || strcmp(line, "quit") == 0) break;
        if (strcmp(line, "clear") == 0 || strcmp(line, "reset") == 0) {
            hist_len = 0;
            printf("[已清空对话历史]\n");
            continue;
        }

        /* 构造完整 prompt: 历史 + [0x01] 新输入 [0x0A] [0x02] */
        uint8_t prompt[HIST_MAX + 1200];
        int p = 0;
        /* 1) 拷贝历史 */
        if (hist_len > 0) {
            memcpy(prompt + p, history, hist_len); p += hist_len;
        }
        /* 2) 追加本轮用户输入 */
        prompt[p++] = 0x01;
        memcpy(prompt + p, line, n); p += n;
        prompt[p++] = 0x0A;
        prompt[p++] = 0x02;

        /* 3) 生成 */
        int ol = 0;
        generate(m, prompt, p, 200, 0.2f, 40, out, &ol);

        /* 4) 打印 */
        printf("AI: ");
        fflush(stdout);
        fwrite(out, 1, ol, stdout);
        printf("\n");

        /* 5) 把本轮 "01...0A 02...0A" 追加到历史 */
        int add = 0;
        /* 5a) 用户部分 */
        if (hist_len + 1 + n + 1 + 1 + ol + 1 > max_hist) {
            /* 空间不够，从最早的部分开始丢弃（整体左移） */
            int need = 1 + n + 1 + 1 + ol + 1;
            int free_space = max_hist - need;
            if (free_space < 0) free_space = 0;
            if (hist_len > free_space) {
                int drop = hist_len - free_space;
                memmove(history, history + drop, hist_len - drop);
                hist_len -= drop;
            }
        }
        /* 5b) 追加用户轮 */
        if (hist_len + 1 + n + 1 <= max_hist) {
            history[hist_len++] = 0x01;
            memcpy(history + hist_len, line, n); hist_len += n;
            history[hist_len++] = 0x0A;
        }
        /* 5c) 追加 AI 轮 */
        if (hist_len + 1 + ol + 1 <= max_hist) {
            history[hist_len++] = 0x02;
            memcpy(history + hist_len, out, ol); hist_len += ol;
            history[hist_len++] = 0x0A;
        }
    }
}

/* =========================================================
 * main
 * ========================================================= */
int main(void) {
    utf8_init();

    printf("==================================================\n");
    printf("  字节级对话机器人  —  C 推理端  (多轮上下文版)\n");
    printf("==================================================\n\n");
    printf("模型文件路径 (回车使用默认 model_int8.btm):\n");
    printf(">>> ");
    fflush(stdout);

    char model_path[512];
    if (!fgets(model_path, sizeof(model_path), stdin)) return 1;

    int n = (int)strlen(model_path);
    while (n > 0 && (model_path[n-1] == '\n' || model_path[n-1] == '\r'))
        model_path[--n] = 0;
    if (n == 0) strcpy(model_path, "model_int8.btm");

    printf("\n[加载] %s\n", model_path);

    Model m;
    if (model_load(model_path, &m) != 0) {
        fprintf(stderr, "[错误] 无法加载模型: %s\n", model_path);
        printf("\n按回车退出...");
        getchar();
        return 2;
    }

    printf("[成功] %s  d=%u  N=%u  h=%u  ff=%u  seq=%u  params=%u\n",
           m.hdr.dtype ? "INT8" : "FP32",
           m.hdr.d_model, m.hdr.n_layers, m.hdr.n_heads,
           m.hdr.d_ff, m.hdr.max_seq, m.total_elems);

    rope_init(m.hdr.d_model / m.hdr.n_heads, m.hdr.max_seq, 10000.0f);
    repl(&m);

    free(m.data);
    free(g_rope_cos);
    free(g_rope_sin);
    return 0;
}
