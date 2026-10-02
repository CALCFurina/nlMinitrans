import json
import os

def convert_jsonl_to_ua(input_path, output_path):
    """
    将 JSONL 转换为自定义的 U:...\nA:... 格式
    用于小模型的 PP / P 阶段训练
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到输入文件: {input_path}")
        return

    seen = set()  # 用于去重
    count = 0     # 有效数据计数

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ 第 {line_num} 行 JSON 解析失败，跳过")
                continue
            
            # 提取字段
            user_text = data.get("prompt", "").strip()
            ai_text = data.get("code", "").strip()
            
            # 跳过空数据
            if not user_text or not ai_text:
                continue
            
            # 核心清洗：把内容里的真实换行符替换为空格，保证结构不破碎
            user_text = user_text.replace("\n", " ").replace("\r", "")
            ai_text = ai_text.replace("\n", " ").replace("\r", "")
            
            # 去重
            pair = (user_text, ai_text)
            if pair in seen:
                continue
            seen.add(pair)
            
            # 编号递增，强制两位数格式 (01., 02., ...)
            count += 1
            f_out.write(f"{count:02d}.\n")
            
            # 写入自定义格式（注意这里是字面量 \n）
            f_out.write(f"U:{user_text}\\nA:{ai_text}\\n\n")

    print(f"✅ 转换完成！")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_path}")
    print(f"   有效对话组: {count} 条")

if __name__ == "__main__":
    # 你可以修改这里的文件名
    INPUT_FILE = "output.jsonl"       # 你的 JSONL 文件
    OUTPUT_FILE = "train.txt"     # 转换后的自定义格式文件
    
    convert_jsonl_to_ua(INPUT_FILE, OUTPUT_FILE)