# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 读取所有数据文件并展示前5行
print("训练集点击日志:")
train_df = pd.read_csv('./data/train_click_log.csv')
print(train_df.head())

print("\n测试集点击日志:")
test_df = pd.read_csv('./data/testA_click_log.csv')
print(test_df.head())

print("\n文章信息:")
articles_df = pd.read_csv('./data/articles.csv')
print(articles_df.head())

print("\n文章Embedding:")
articles_emb_df = pd.read_csv('./data/articles_emb.csv')
print(articles_emb_df.head())

# 从训练集中随机划分50,000用户作为新的测试集
# 获取所有唯一用户ID
all_users = train_df['user_id'].unique()

# 随机选择50000个用户
np.random.seed(42)  # 设置随机种子以确保结果可复现
# np.random.seed(42)：设置 numpy 随机数生成器的种子为 42
test_users = np.random.choice(all_users, size=50000, replace=False)
# np.random.choice(all_users, ...)：从all_users数组中随机选择元素
# replace=False：设置为 “不放回抽样”，确保每个用户只被选中一次（测试集中的用户不重复）

# 将选中用户的交互作为测试集
new_test_df = train_df[train_df['user_id'].isin(test_users)]
# train_df['user_id'].isin(test_users)：判断train_df中每行的user_id是否在test_users列表中，返回布尔值数组（True表示该用户是测试用户）
new_train_df = train_df[~train_df['user_id'].isin(test_users)]
# ~：逻辑非运算符，反转布尔值（True变False，False变True）
# train_df[~...]：筛选出所有不是测试用户的交互记录，作为新的训练集

print(f"新训练集大小: {len(new_train_df)}")
print(f"新测试集大小: {len(new_test_df)}")
print(f"新训练集用户数: {new_train_df['user_id'].nunique()}")
print(f"新测试集用户数: {new_test_df['user_id'].nunique()}")

# articles_df和articles_emb_df仅保留出现在训练集和测试集中的item
# 获取训练集和测试集中出现的所有文章ID
train_items = set(new_train_df['click_article_id'].unique())
# unique()会返回一个 numpy 数组，包含 click_article_id 列中所有不重复的文章 ID
test_items = set(new_test_df['click_article_id'].unique())
all_items = train_items | test_items # 使用集合的并集操作（|），将训练集和测试集中的文章 ID 合并

# 仅保留这些文章的信息
articles_df = articles_df[articles_df['article_id'].isin(all_items)]
articles_emb_df = articles_emb_df[articles_emb_df['article_id'].isin(all_items)]

print(f"保留的文章数: {len(articles_df)}")
print(f"保留的文章embedding数: {len(articles_emb_df)}")

# 导入所需的库
import math
from tqdm import tqdm
from collections import defaultdict
import pickle
import os

# 用两种方式计算物品间的相似度
    # 基于协同过滤（同上一节）
    # 基于内容embedding
def get_user_item_time(df):
    """
    构建用户-物品-时间交互字典

    Args:
        df: 包含用户点击记录的数据框,包含user_id,click_article_id,click_timestamp等字段

    Returns:
        dict: 用户交互历史字典 {user_id: [(item_id, timestamp), ...]}
    """
    user_item_time_dict = defaultdict(list)

    for row in df.itertuples():
        # 遍历数据框的每一行（itertuples()返回每行数据的命名元组）
        user_item_time_dict[row.user_id].append((row.click_article_id, row.click_timestamp))

    # 按时间戳排序,保证时间顺序
    for user_id in user_item_time_dict:
        user_item_time_dict[user_id].sort(key=lambda x: x[1])

    return user_item_time_dict


def itemcf_sim(df):
    """
    计算物品协同过滤的相似度矩阵

    Args:
        df: 用户点击历史数据框

    Returns:
        dict: 物品相似度矩阵 {item_id: {item_id: similarity_score, ...}, ...}

    计算步骤:
    1. 获取用户-物品-时间交互字典
    2. 遍历每个用户的历史记录,统计物品共现次数
    3. 使用余弦相似度对共现次数进行归一化
    4. 考虑用户历史序列长度的惩罚项
    """

    # 1. 获取用户交互历史
    user_item_time_dict = get_user_item_time(df)

    # 2. 计算物品相似度
    i2i_sim = {}  # 物品-物品共现矩阵
    item_cnt = defaultdict(int)  # 物品被点击次数

    print("开始计算物品相似度矩阵...")
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 遍历用户的每个交互物品
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})

            # 遍历同一用户点击的其他物品
            for j, j_click_time in item_time_list:
                if i == j:
                    continue

                # 初始化物品j的相似度
                i2i_sim[i].setdefault(j, 0)

                # 计算相似度,考虑用户的历史序列长度作为惩罚项
                # 序列越长,物品间相似度越小
                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)

    # 3. 对相似度进行余弦归一化
    i2i_sim_ = i2i_sim.copy()
    print("开始相似度归一化...")
    for i, related_items in tqdm(i2i_sim.items()):
        for j, wij in related_items.items():
            # 使用余弦相似度公式: wij / sqrt(|Ui| * |Uj|)
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # 4. 保存相似度矩阵
    save_path = './data/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))

    return i2i_sim_


def itememb_sim(articles_emb_df):
    """
    基于文章embedding计算相似度矩阵

    Args:
        articles_emb_df: 文章embedding数据框，包含article_id和emb_0~emb_249

    Returns:
        dict: 物品相似度矩阵 {item_id: {item_id: similarity_score, ...}, ...}

    计算步骤:
    1. 将embedding数据转换为numpy数组
    2. 对embedding进行归一化
    3. 计算余弦相似度矩阵
    4. 转换为字典格式并保存
    """

    print("开始计算文章embedding相似度矩阵...")

    # 1. 准备embedding数据
    article_ids = articles_emb_df['article_id'].values  # 返回物品ID列表（numpy数组）
    article_embs = articles_emb_df.iloc[:, 1:].values  # .iloc()允许你通过行号和列号来选择数据，而不是通过行名或列名
    # 返回嵌入向量矩阵Numpy数组（排除第一列article_id）

    # 2. 对embedding进行L2归一化（使向量模长为1，此时内积=余弦相似度）
    from sklearn.preprocessing import normalize
    article_embs = normalize(article_embs, axis=1)
    # axis=0 表示沿着列的方向进行操作，即对每一列进行归一化。
    # axis=1 表示沿着行的方向进行操作，即对每一行进行归一化

    # 3. 批量计算相似度矩阵(直接用矩阵乘法计算内积)（避免一次性计算全量矩阵导致内存溢出）
    print("计算内积相似度...")
    sim_matrix = np.zeros((len(article_embs), len(article_embs)))
    # NumPy 数组的操作是通过预编译的 C 代码实现的，因此执行速度非常快。NumPy 提供了高效的数组操作，如向量化计算，避免了显式循环
    # NumPy 数组在内存中是连续存储的，每个元素直接存储在数组中，没有额外的指针开销
    # Python 列表是动态数组，每个元素都是一个指向对象的指针。这意味着列表在内存中存储的是指向实际对象的引用，而不是对象本身
    batch_size = 16  # 每次计算的批量大小（可根据内存调整）
    for i in tqdm(range(0, len(article_embs), batch_size)):
        batch_end = min(i + batch_size, len(article_embs)) # 计算当前批次的结束索引：如果i + batch_size超过文章总数，则取文章总数作为结束索引
        # 直接用矩阵乘法计算内积
        batch_sims = np.dot(article_embs[i:batch_end], article_embs.T)
        # numpy.dot 函数的返回值是一个 NumPy 数组
        # article_embs[i:batch_end]是当前批次的嵌入向量（形状为(batch_size, 250)）
        # article_embs.T是所有文章嵌入向量的转置（形状为(250, 文章数量)）
        sim_matrix[i:batch_end] = batch_sims
        # Numpy数组可以块赋值
    # 4. 转换为字典格式
    print("转换格式...")
    i2i_sim = {}
    for i, item_id in enumerate(tqdm(article_ids)):
        i2i_sim[item_id] = {}
        # 只保存相似度最高的100个物品
        sorted_idx = np.argsort(sim_matrix[i])[::-1][1:101]  # 除去自身  自己和自己肯定相似度最高 是1
        # 由于 step=-1 是反向切片，Python 会默认 start 为最后一个元素的索引（len(数组)-1），end 为第一个元素的前一位 看冒号作为分隔符[start:end:step]
        # np.argsort 是 NumPy 库中的一个函数，用于返回数组值从小到大的索引值。它不改变原数组的顺序，而是返回一个新数组，其中包含原数组中元素的排序索引
        for j in sorted_idx:
            i2i_sim[item_id][article_ids[j]] = sim_matrix[i][j]

    # 5. 保存相似度矩阵
    save_path = './data/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pickle.dump(i2i_sim, open(save_path + 'itemembedding_i2i_sim.pkl', 'wb'))

    print(f"相似度矩阵计算完成，共包含{len(i2i_sim)}篇文章")
    return i2i_sim

# 计算基于协同过滤的相似度矩阵
i2i_sim = itemcf_sim(new_train_df)
# 计算基于内容embedding的相似度矩阵
i2i_embsim = itememb_sim(articles_emb_df)
# 对比两种相似度矩阵

# 计算每个item在两种相似度下的相似item集合的Jaccard相似度
print("计算两种相似度矩阵的Jaccard相似度...")
jaccard_sims = []
common_items = set(i2i_sim.keys()) & set(i2i_embsim.keys())  # sim是共现过即相似 embsim是相似度top100

for item in tqdm(common_items):
    # 获取两种方法下物品的相似item集合
    cf_sims = set(i2i_sim[item].keys())
    emb_sims = set(i2i_embsim[item].keys())

    # 计算Jaccard相似度
    intersection = len(cf_sims & emb_sims) #交
    union = len(cf_sims | emb_sims) # 并
    jaccard_sim = intersection / union if union > 0 else 0
    jaccard_sims.append(jaccard_sim)  # 由于common_items是set存储 故是无序的

# 绘制Jaccard相似度分布图
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(jaccard_sims, bins=50, edgecolor='black')
# plt.hist 用于绘制直方图 显示数值数据的分布情况 bins：如果是整数，表示直方图的柱数 如果是序列，表示直方图的边界 edgecolor：柱的边缘颜色
plt.title('Distribution of Jaccard Similarity between CF and Embedding Similar Items')
plt.xlabel('Jaccard Similarity')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

# 打印一些统计信息
print(f"\n统计信息:")
print(f"平均Jaccard相似度: {np.mean(jaccard_sims):.4f}")
print(f"中位数Jaccard相似度: {np.median(jaccard_sims):.4f}")
print(f"最大Jaccard相似度: {np.max(jaccard_sims):.4f}")
print(f"最小Jaccard相似度: {np.min(jaccard_sims):.4f}")
