import os, json

PROMPT_DIR = "/home/jiaji_lu/AR/Infinity/evaluation/DPG/prompts"
OUT_FILE = "DPG_prompts.jsonl"

txt_files = [fn for fn in os.listdir(PROMPT_DIR) if fn.lower().endswith(".txt")]
total = len(txt_files)
print(f"total {total} file")

success = 0
fail = 0

with open(OUT_FILE, "w", encoding="utf-8") as fout:
    for fn in txt_files:
        path = os.path.join(PROMPT_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                prompt = f.readline().strip()
            if not prompt:
                raise ValueError("empty prompt")
            json.dump({"prompt": prompt}, fout, ensure_ascii=False)
            fout.write("\n")
            success += 1
        except Exception as e:
            print(f"convert fail:{fn}, because:{e}")
            fail += 1

print(f"\n finished convert: success {success}/{total}, fail: {fail}/{total}")
