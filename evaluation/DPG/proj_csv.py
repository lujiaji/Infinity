import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="根据 metadata 更新 dpg_bench.csv 中的 item_id")
    parser.add_argument('--bench_csv', type=str, default='/home/jiaji_lu/AR/ELLA/dpg_bench/dpg_bench.csv',
                        help='原始的 dpg_bench.csv 文件路径')
    parser.add_argument('--meta_csv', type=str, default='/home/jiaji_lu/AR/Infinity/evaluation/DPG/dpg_metadata.csv',
                        help='生成的 DPGdpg_metadata.csv 文件路径')
    parser.add_argument('--output_csv', type=str, default='/home/jiaji_lu/AR/Infinity/evaluation/DPG/dpg_bench_updated.csv',
                        help='输出更新后 CSV 的路径')
    args = parser.parse_args()

    # 读取 metadata，构建 text->item_id 的映射
    meta = pd.read_csv(args.meta_csv, dtype=str)
    mapping = dict(zip(meta['text'], meta['item_id']))

    # 读取原始 dpg_bench.csv
    bench = pd.read_csv(args.bench_csv, dtype=str)

    # 替换 item_id：若 text 在映射中，则使用新的 item_id；否则保留原 id
    bench['item_id'] = bench['text'].map(mapping).fillna(bench['item_id'])

    # 保存为新文件
    bench.to_csv(args.output_csv, index=False, encoding='utf-8')
    print(f"✅ 已保存更新后的 CSV：{args.output_csv}")

if __name__ == '__main__':
    main()
