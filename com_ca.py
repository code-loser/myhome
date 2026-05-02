import pandas as pd
import matplotlib.pyplot as plt
import re

def get_processed_data(file_path):
    """
    辅助函数：读取并处理CSV文件，返回处理好的Series
    """
    try:
        # 读取CSV，假设无表头
        df = pd.read_csv(file_path, header=None, names=['layer_str', 'col2', 'value'])
        # 提取数字索引
        df['layer_idx'] = df['layer_str'].apply(lambda x: int(re.search(r'\d+', str(x)).group()))
        # 分组求平均并排序
        return df.groupby('layer_idx')['value'].mean().sort_index()
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

# --- 主绘图程序 ---

plt.figure(figsize=(12, 7))

# 1. 处理并绘制第一个CSV文件 (假设文件名为 data1.csv)
data1 = get_processed_data('debug_moe_ratios1.csv')
if data1 is not None:
    plt.plot(data1.index, data1.values, marker='o', linestyle='-', 
             color='tab:blue', label='low_precision experts ratio when T1=0.3')

# 2. 处理并绘制第二个CSV文件 (假设文件名为 data2.csv)
data2 = get_processed_data('debug_moe_ratios2.csv')
if data2 is not None:
    plt.plot(data2.index, data2.values, marker='s', linestyle='-', 
             color='tab:green', label='low_precision experts ratio when T1=0.6')

# 3. 绘制第一条理论线 y=0.7
plt.axhline(y=0.7, color='tab:blue', linestyle='--', linewidth=1.5, label='low_precision experts ratio Theory Line (y=0.7)')

# 4. 绘制第二条理论线 y=0.3
plt.axhline(y=0.4, color='tab:green', linestyle='--', linewidth=1.5, label='low_precision experts ratio Theory Line (y=0.4)')

# --- 图表美化 ---
plt.title('Comparison of Experimental Data and Theoretical Baselines', fontsize=14)
plt.xlabel('Layer Index', fontsize=12)
plt.ylabel('Average Value (Column 3)', fontsize=12)

# 设置坐标轴范围（可选，根据你的数据范围调整）
# plt.ylim(0, 1.0) 

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best', frameon=True) # 显示图例

# 保存并展示
plt.tight_layout()
plt.savefig('combined_comparison_plot.png', dpi=300)
print("合图已保存为: combined_comparison_plot.png")
plt.show()