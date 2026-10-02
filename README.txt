nlMinitrans --A pure-Lua Chinese Transformer on TI-Nspire 
nlMinitrans --一个在TI-Nspire上的纯Lua的中文Transformer


nl  = TI-Nspire Lua

Mini = A tiny model/一个微模型

trans = It is a Transformer-based Language Model./这是一个基于Transformer的语言模型


It's a Language Model Running on the TI-Nspire(CX/CXII),and only the TI-NspireCX/CXII can use the largest context window(1024 bytes).
这是一个在TI-Nspire(CX/CXII)运行的语言模型，只有 TI-Nspire CX/CXII才能使用最大上下文窗口（1024 字节）。

It's SO slow ,the Time to First Byte is around 40s,and the Bytes per second is around 0.5Byte/s.(Yeah,it is a byte-level language model,so if she wants to reply one Chinese character,she has to output 3 consecutive bytes(The character encoding on the TI-Nspire is UTF-8.))
它的速度非常慢：首字节耗时约 40 秒，生成速度大约 0.5 字节/秒。（没错，这是一个字节级语言模型，所以如果它要输出一个汉字，就需要连续输出 3 个字节。(TI-Nspire 上使用 UTF-8 字符编码。)）

To train a model,execute the "main.py" and input your training dataset filenames(训练文件名,supports sequential training over multiple datasets) and epochs(轮次),then the model will start training(if a "model_fp32.btm" in the output folder(输出目录) ,training will resume from the weights stored inside this file.)
训练模型：运行"main.py"，输入训练数据集文件名（支持多数据集依次训练）与训练轮次，随后模型开始训练。若输出目录中存在"model_fp32.btm"文件，则会从该文件内保存的权重继续断点续训。

Note:if your training datasets is too long,you can use the "cut.py" to split the dataset
注：数据集太长可以用"cut.py"切分

Note2: The format for the training datasets is like this/格式如下:


01.
U:(User says/用户说的)\nA:(Model says/模型说的)\n
U:(User says/用户说的)\nA:(Model says/模型说的)\n
......

02.
......

......



To package a model,execute the "bin2weight.py" to convert the "model_int8.btm" model to a lua-embedded codes in "weights.lua",then copy the code in the and replace the model part in the "transBW.lua".Finally,pack it to ".tns" with a tool such as Tns Tool.
包装模型，先执行"bin2weight.py"来把"model_int8.btm"转换成存进"weights.lua"里面的可以嵌入lua代码的代码，然后复制代码，粘贴到"transBW.lua"的模型区。最后，使用像Tns tool等的工具来包装成tns

Its weight file is only 88KB(INT8 quantization),so she can only generate some incoherent Chinese sentences and she is silly.
权重只有88KB(INT8量化)，所以她只会生成一些不通顺的中文句子，而且她很傻

(Most of the source code is generated with assistance from the Big Blue Killer Whale(Deepseek-V4.1 Flash))
(绝大多数代码由蓝色大肥鱼(Deepseek-V4.1 Flash)生成)

## License & Dataset Attribution/许可证与数据集归属
This project is released under the LGPL v3.0
本项目基于 GNU LGPL v2.1 许可证发布。

Training datasets used:
1. PPT(Pre-Pre-Training datasets): 预训练数据集,https://www.modelscope.cn/datasets/Rentuiteam/Train_datasets/files,Apache License 
2. PT (Pre-Training datasets): 沐雪（中文角色扮演）训练集,https://www.modelscope.cn/datasets/Moemuu/Muice-Dataset/files,CC-BY-NC-4.0
3. P.5T: 多轮对话聊天数据,https://www.modelscope.cn/datasets/justgo10000/Multi-turn-dialogue/files,Apache License 2.0

All training text data retains the original copyright and license terms of their respective authors.
If you redistribute this project, you must comply with each dataset's original license.

