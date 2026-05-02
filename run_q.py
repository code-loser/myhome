import json

import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from transformers import AutoConfig, AutoTokenizer, TextStreamer
import time
import math
import argparse
from transformers import AutoProcessor

# 假设你的 src 目录下的 build_model 已经支持了 Qwen 的架构
# 注意：如果 src/build_model.py 内部硬编码了 Mixtral 的层名称，你可能需要修改该文件
from src1.build_model import OffloadConfig, build_model
from src1.dp import get_cache_size
from tqdm import tqdm # 推荐引入进度条，评测过程不枯燥
from modelscope.msdatasets import MsDataset

def main(args):
    # 1. 设置模型路径
    path = "/data1/zj_models/HL-MOE"
    model_name = path
    state_path = path

    # 2. 获取配置，解决 AttributeError
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Qwen3-VL 的层数获取逻辑（尝试多种可能属性）
    num_layers = getattr(config, "num_hidden_layers", 
                         getattr(config, "num_layers", 
                                 getattr(config.text_config, "num_hidden_layers", 32)))
    
    # 获取专家总数（Qwen 的属性名通常是 moe_intermediate_size 相关的配置，或者是 num_experts）
    # 这里建议打印 config 确认一下
    num_experts = getattr(config, "num_experts", 
                          getattr(config.text_config, "num_experts", 64)) 

    device = torch.device(f"cuda:{args.device}")

    # 3. 卸载配置 (Offload Config)
    # 移除了 adapgate 相关逻辑，直接设置缓存策略
    main_size = args.size 
    cache_strategy = get_cache_size(main_size, False) # 强制关闭 adapgate 逻辑

    offload_config = OffloadConfig(
        main_size=main_size,
        cache_strategy=cache_strategy,
        offload_size = 2 * num_experts, # 动态适配专家总数
        buffer_size=6,
    )

    # 5. 构建模型
    # 重要：build_model 函数必须内部兼容 Qwen 的 Module 层级结构
    model = build_model(
        device=device,
        offload_config=offload_config,
        state_path=state_path,
    )

    # 移除了所有关于 weight[idx] 和 threshold 的代码块

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    past_key_values = None
    total_time = 0
    
    print("Model loaded. Ready for inference.")
    print("Model loaded. Ready for inference.")
    # run_mmvet_evaluation(model, model_name, device) # 直接调用 MM-Vet 评测函数
    # # --- 1. 加载 ModelScope 数据集 ---
    # print("Loading MMBench dataset from ModelScope...")
    # # 使用 dev 划分来获取带有真值的答案进行准确率验证
    # ds = MsDataset.load('lmms-lab/MMBench', subset_name='en', split='dev')

    # correct = 0
    # total = 0
    # total_generated_tokens = 0
    # total_generation_time = 0.0
    # processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    # print("\n" + "="*20 + " 开始 MMBench 推理评测 " + "="*20)

    # # 使用 tqdm 包装，直观看到进度
    # for i, item in enumerate(tqdm(ds, desc="Evaluating")):
    #     # 1. 拼接问题与选项 (A, B, C, D)
    #     if i>0:
    #         break
    #     question = item['question']
    #     options = ""
    #     for opt in ['A', 'B', 'C', 'D']:
    #         if opt in item and item[opt] is not None:
    #             options += f"\n{opt}. {item[opt]}"
        
    #     # 构造最终 Prompt：明确要求只输出选项字母
    #     prompt_text = f"{question}{options}\nAnswer with the option letter from the given choices directly."
    #     image = item['image'] # PIL Image

    #     # 2. 构造 Qwen3-VL 标准的消息模板
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image", "image": image},
    #                 {"type": "text", "text": prompt_text},
    #             ],
    #         }
    #     ]
        
    #     # 3. 使用原生 processor 处理多模态输入
    #     # 注意：这里假设你外层初始化的是 processor = AutoProcessor.from_pretrained(...)
    #     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

    #     start_time = time.time()
        
    #     # 4. 执行你自定义引擎的推理
    #     with torch.inference_mode():
    #         # 注意：评测客观题时，关闭 do_sample，max_new_tokens设小一点即可
    #         result = model.generate(
    #             **inputs, # 将 input_ids, attention_mask, pixel_values 等一并传入
    #             max_new_tokens=10, 
    #             pad_token_id=processor.tokenizer.eos_token_id,
    #             do_sample=False, # 🌟 关键：学术评测必须用 Greedy Decoding 保证结果确定性
    #             return_dict_in_generate=True,
    #         )
        
    #     end_time = time.time()
    #     generation_time = end_time - start_time
    #     total_generation_time += generation_time
        
    #     # 5. 结果解码与统计
    #     # 截取模型新生成的 token
    #     input_len = inputs.input_ids.shape[1]
    #     generated_ids = result.sequences[0, input_len:]
    #     num_generated = len(generated_ids)
    #     total_generated_tokens += num_generated
        
    #     pred_text = processor.decode(generated_ids, skip_special_tokens=True).strip().upper()
    #     # 简单的答案提取逻辑：取模型输出的第一个字母
    #     pred_answer = pred_text[0] if len(pred_text) > 0 else ""
        
    #     # 获取真值
    #     gt_answer = str(item.get('answer', '')).strip().upper()
        
    #     if gt_answer:
    #         total += 1
    #         if pred_answer == gt_answer:
    #             correct += 1
                

    # # --- 6. 最终结果结算 ---
    # if total > 0:
    #     print("\n" + "="*20 + " 评测报告 " + "="*20)
    #     print(f"Final Accuracy: **{correct/total:.2%}** ({correct}/{total})")
        
    #     if total_generation_time > 0:
    #         avg_speed = total_generated_tokens / total_generation_time
    #         print(f"Average Decoding Speed: **{avg_speed:.2f} tokens/s**")
    # else:
    #     print("\nInference finished. (No ground truth found for this split)")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        # 针对 VL 模型的模板处理（如果是纯文本对话）
        messages = [{"role": "user", "content": user_input}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

        # 简单的 Attention Mask 生成
        attention_mask = torch.ones_like(input_ids)

        print("Qwen3-VL: ", end="")
        start_time = time.time()
        
        with torch.inference_mode():
            result = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                streamer=streamer,
                do_sample=True,
                top_k=4,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # 假设生成了 tokens 数量
        num_generated = result.sequences.shape[1] - input_ids.shape[1]
        if num_generated > 0:
            print(f"\n[Speed: {num_generated / generation_time:.2f} tokens/s]")
def run_mmvet_evaluation(model ,model_name, device):
    # 1. 加载数据集 (MM-Vet)
    output_json_path = "mmvet_textfulldis.json"
    print("正在加载 MM-Vet 数据集...")
    ds =  MsDataset.load('lmms-lab/MMVet', subset_name='default', split='test')
    
    # 2. 初始化模型组件
    # 假设 model 已经在你的环境中初始化好了，或者是你自定义的推理引擎
    # model = YourCustomMoEModel.from_pretrained(model_name).to(device) 
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    results = {}
    total_generated_tokens = 0
    total_generation_time = 0.0
    
    print("\n" + "="*20 + " 开始 MM-Vet 推理评测 " + "="*20)
    print(f"评测样本数: {len(ds)}")

    # 使用 tqdm 包装
    for i, item in enumerate(tqdm(ds, desc="MM-Vet Generating")):
        question_id = item['question_id']
        question = item['question']
        image = item['image'] # PIL Image
        # 3. 构造开放式 Prompt
        # MM-Vet 不需要选项，直接问问题即可。为了效果更好，可以稍微引导模型详细回答。
        prompt_text = f"{question}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # 4. 前向处理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
        token_names = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    
        # 判定 Token 类型
        token_types = [
            "Image" if any(tag in name for tag in ["<|image_pad|>", "<|visual|>", "patch"]) else "Text"
            for name in token_names
        ]

        vision_mask = torch.tensor(
            [t == "Image" for t in token_types], 
            dtype=torch.bool, 
            device=device  # 极其重要：必须和你的计算张量在同一个显卡上
        )
        # 如果你的 inputs 是 2D 的 (batch_size, seq_len)，你可能需要 reshape 一下
        # vision_mask = vision_mask.view(inputs.shape)
        for i in range(48):
            layer = model.model.language_model.layers[i]
            layer.mlp.token_type = vision_mask
        # 5. 执行推理
        start_time = time.time()
        with torch.inference_mode():
            # 🌟 关键参数修改：
            # - max_new_tokens: 设为 512 以保证逻辑推理链完整
            # - do_sample: False 保证实验可重复性
            result = model.generate(
                **inputs,
                max_new_tokens=512, 
                pad_token_id=processor.tokenizer.eos_token_id,
                do_sample=False, 
                return_dict_in_generate=True,
            )
        end_time = time.time()

        # 6. 统计与解码
        generation_time = end_time - start_time
        input_len = inputs.input_ids.shape[1]
        generated_ids = result.sequences[0, input_len:]
        
        num_generated = len(generated_ids)
        total_generated_tokens += num_generated
        total_generation_time += generation_time
        
        # 解码完整文本作为结果
        pred_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        # 将结果存入字典，Key 为 question_id
        results[question_id] = pred_text

        # 调试打印 (可选，只打印前几个)
        if i < 2:
            print(f"\nID: {question_id} | Output: {pred_text[:100]}...")

    # 7. 保存结果为 MM-Vet 官方要求的格式
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # 8. 总结报告
    print("\n" + "="*20 + " 推理完成 " + "="*20)
    print(f"结果已保存至: {output_json_path}")
    if total_generation_time > 0:
        avg_speed = total_generated_tokens / total_generation_time
        print(f"平均推理速度: **{avg_speed:.2f} tokens/s**")
        print(f"平均每个问题生成长度: {total_generated_tokens/len(ds):.1f} tokens")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=60, help="Main cache size")
    # 即使不使用，也保留参数位防止原脚本报错，或者直接清理
    parser.add_argument("--adapgate", action="store_true", help="Disabled for Qwen3-VL") 
    parser.add_argument("--device", default=1, help="GPU device ID") 
    args = parser.parse_args()
    main(args)