import pandas as pd
import matplotlib.pyplot as plt
import re

def plot_expert_loading(file_path):
    # 1. 读取 CSV 文件
    # 由于 torch.Size([187, 8]) 内部含有逗号，默认读取会产生多余列
    # 我们不指定表头，直接按位置索引读取
    try:
        df_raw = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return

    # 2. 提取数据
    # 根据描述：第一列是层名，最后三列是数据
    df = pd.DataFrame()
    
    # 提取 Layer ID (从 "Layer_0" 中提取数字 0)
    df['layer_id'] = df_raw.iloc[:, 0].apply(lambda x: int(re.search(r'\d+', str(x)).group()))
    
    # 提取后三列数值
    df['Hobbit'] = pd.to_numeric(df_raw.iloc[:, -3])
    df['Baseline'] = pd.to_numeric(df_raw.iloc[:, -2])
    df['My'] = pd.to_numeric(df_raw.iloc[:, -1])

    # 3. 处理多样本：按 layer_id 分组取平均值
    avg_df = df.groupby('layer_id').mean().sort_index().reset_index()

    # 4. 绘图
    plt.figure(figsize=(10, 6))
    
    # 画三条折线
    plt.plot(avg_df['layer_id'], avg_df['Hobbit'], marker='o', linestyle='-', label='MoE-APEX')
    plt.plot(avg_df['layer_id'], avg_df['Baseline'], marker='s', linestyle='--', label='Baseline (Full HP)')
    # plt.plot(avg_df['layer_id'], avg_df['My'], marker='^', linestyle='-', label='our Method (Mixed Precision)')

    # 图表修饰
    plt.title('Comparison of Expert Data Loading Volume per Layer in prefill', fontsize=14)
    plt.xlabel('Layer ID', fontsize=12)
    plt.ylabel('Data Volume (Units of x bits)', fontsize=12)
    plt.xticks(avg_df['layer_id']) # 确保显示所有层 ID
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()
    
    # 优化布局并保存/显示
    plt.tight_layout()
    plt.savefig('expert_loading_comparison_moti.png', dpi=300)
    plt.show()

    # 打印平均值表格以便检查
    print("Averaged Data per Layer:")
    print(avg_df)

# 使用示例
if __name__ == "__main__":
    # 请确保你的文件名正确
    plot_expert_loading('debug_moe_ratios.csv')