def bin_to_byte_list(file_path):
    """读取 bin 文件，返回一字节一元素的整数列表"""
    with open(file_path, 'rb') as f:
        data = f.read()
    return list(data)   # bytes 对象迭代出来就是 0~255 的整数


def main():
    file_path = input("请输入 bin 文件路径: ").strip().strip('"').strip("'")

    try:
        byte_list = bin_to_byte_list(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        input("\n按 Enter 键退出...")
        return
    except IsADirectoryError:
        print(f"错误：{file_path} 是一个目录，不是文件")
        input("\n按 Enter 键退出...")
        return
    except PermissionError:
        print(f"错误：没有权限读取 {file_path}")
        input("\n按 Enter 键退出...")
        return

    # 拼接成 "[111,0,255]" 格式
    result = "[" + ",".join(str(b) for b in byte_list) + "]"
    print(result)

    input("\n按 Enter 键退出...")


if __name__ == "__main__":
    main()