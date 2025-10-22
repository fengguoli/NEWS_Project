# 导入必要的库
import pandas as pd
# pandas擅长于将表格数据转化为dataframe格式处理
import numpy as np
# 导入数值计算库numpy，它提供了高性能的多维数组（ndarray）和数学函数，是pandas等库的底层依赖，常用于处理数值型数据。
import matplotlib.pyplot as plt
# 导入matplotlib库中的pyplot模块，matplotlib是 Python 的基础可视化库，pyplot模块提供了类似 MATLAB 的绘图接口，可用于绘制折线图、柱状图、散点图等
import seaborn as sns
# 导入基于matplotlib的高级可视化库seaborn，它默认提供了更美观的图表样式，且简化了复杂图表（如热图、分布图）的绘制流程
sns.set_theme()
# 指定使用seaborn风格的图表（如网格线、配色等），比默认风格更美观，适合数据分析展示。
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams：matplotlib的配置参数字典，用于修改全局绘图设置 'font.sans-serif'：表示设置无衬线字体（适合屏幕显示的字体类型）
# ['SimHei']：将字体设置为SimHei（黑体），解决中文在图表中显示为方块或乱码的问题（默认字体不支持中文）。
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 'axes.unicode_minus'：matplotlib中控制负号显示的参数 False：关闭默认的 Unicode 负号渲染方式，避免负号在中文环境下显示为方块（配合上面的字体设置，确保负号正常显示为-）

# 1. 读取所有数据文件并展示前5行
print("训练集点击日志:")
train_df = pd.read_csv('./data/train_click_log.csv')
# pd.read_csv()：pandas读取 CSV 文件的函数，返回一个DataFrame（表格型数据结构）
print(train_df.head())
# display()：在 Jupyter Notebook 或 IPython 环境中用于展示对象的函数，比print()更美观（会以表格形式展示DataFrame）
# train_df.head()：DataFrame的head()方法，默认返回前 5 行数据（可通过参数指定行数，如head(10)返回前 10 行）。
# 作用：快速预览训练集的字段（列名）和部分数据，了解数据格式（如是否有用户 ID、文章 ID、点击时间等）。

print("\n测试集点击日志:")
test_df = pd.read_csv('./data/testA_click_log.csv')
print(test_df.head())

print("\n文章信息:")
articles_df = pd.read_csv('./data/articles.csv')
print(articles_df.head())

print("\n文章Embedding:")
articles_emb_df = pd.read_csv('./data/articles_emb.csv')
print(articles_emb_df.head())

# 2. 统计训练集和测试集的基本信息
print("训练集统计:")
print(f"用户数量: {train_df['user_id'].nunique():,}") # 统计训练集中有多少个不同的用户
# .nunique()：pandas的 Series 方法，用于计算该列中不重复值的数量（即去重后的用户总数）。
# :,：格式化符号，用于在数字中添加千位分隔符（如 10000 显示为 10,000），使结果更易读。
print(f"文章数量: {train_df['click_article_id'].nunique():,}") # 了解训练集涉及多少篇不同的文章
print(f"点击数量: {len(train_df):,}") # 反映训练集的样本量大小（总交互记录数）
# len(train_df)：len()函数返回DataFrame的行数，由于每一行代表一次用户点击行为，因此行数即总点击次数。

print("\n测试集统计:")
print(f"用户数量: {test_df['user_id'].nunique():,}")
print(f"文章数量: {test_df['click_article_id'].nunique():,}")
print(f"点击数量: {len(test_df):,}")

# 3. 分析训练集和测试集的重叠情况，是否存在cold start问题
# 用户重叠分析
train_users = set(train_df['user_id'].unique())
# train_df['user_id'].unique()：提取训练集中所有不重复的用户 ID，返回一个 numpy 数组（包含训练集所有用户）pandas Series的unique()保持首次出现顺序
# set(...)：将数组转换为集合（set），集合的特性是元素唯一且支持集合运算（如交集、差集）
test_users = set(test_df['user_id'].unique())
overlap_users = train_users & test_users
new_users = test_users - train_users # -：集合的差集运算符，计算只在测试集中存在、不在训练集中出现的元素。
# new_users：存储测试集中的 “新用户”（训练集中没有其行为数据），这类用户可能导致冷启动问题。

# 文章重叠分析
train_items = set(train_df['click_article_id'].unique())
test_items = set(test_df['click_article_id'].unique())
overlap_items = train_items & test_items
new_items = test_items - train_items


print("用户重叠分析:")
print(f"训练集用户数: {len(train_users):,}")
print(f"测试集用户数: {len(test_users):,}")
print(f"重叠用户数: {len(overlap_users):,}")
print(f"测试集中新用户数: {len(new_users):,}")

print("\n文章重叠分析:")
print(f"训练集文章数: {len(train_items):,}")
print(f"测试集文章数: {len(test_items):,}")
print(f"重叠文章数: {len(overlap_items):,}")
print(f"测试集中新文章数: {len(new_items):,}")
