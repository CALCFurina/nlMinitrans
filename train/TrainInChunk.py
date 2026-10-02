# train_all_chunks.py
import os
import glob
from main import train, export_btm

CHUNK_DIR = "train_chunks"
EPOCHS_PER_CHUNK = 3

def main():
    chunks = sorted(glob.glob(os.path.join(CHUNK_DIR, "train_*.txt")))
    print(f"[chunk] 共 {len(chunks)} 块")

    model = None
    for i, path in enumerate(chunks):
        print(f"\n========== 块 {i+1}/{len(chunks)}: {os.path.basename(path)} ==========")
        model = train(path, model=model, epochs=EPOCHS_PER_CHUNK, lr=1e-3)
        export_btm(model, f"pp_chunk_{i:02d}.btm", "fp32")
        print(f"[save] pp_chunk_{i:02d}.btm")

    export_btm(model, "model_fp32.btm", "fp32")
    export_btm(model, "model_int8.btm", "int8")
    print("[done] 全部块训练完成")

if __name__ == "__main__":
    main()