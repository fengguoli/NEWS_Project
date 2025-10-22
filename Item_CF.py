# 导入所需的库
import pandas as pd
import Data_analysis
import math
# 导入 Python 内置的math模块，该模块提供了各种数学运算函数（如三角函数、对数、指数等）
from tqdm import tqdm
# 从tqdm库中导入tqdm函数，该函数用于在循环中显示进度条，方便观察任务执行进度
from collections import defaultdict
# 与普通字典不同，defaultdict会为不存在的键自动创建默认值（本代码中默认值为列表），避免直接访问不存在的键时出现KeyError错误，非常适合构建嵌套结构的字典。
# dd = defaultdict(int)        # 默认值 0
# dl = defaultdict(list)       # 默认值 []
# ds = defaultdict(set)        # 默认值 set()
# dd2 = defaultdict(dict)      # 默认值 {}
import pickle
# 导入pickle模块，该模块用于将 Python 对象（如字典、列表）序列化（保存到文件）或反序列化（从文件加载)
import os
# 导入os模块，该模块提供了与操作系统交互的功能（如文件路径处理、创建目录等）

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
        # df.itertuples()：返回一个迭代器，每次迭代生成一个包含该行数据的命名元组（可以通过字段名访问具体值，如row.user_id）
        user_item_time_dict[row.user_id].append((row.click_article_id, row.click_timestamp)) # 列表里面装的是元组

    # 按时间戳排序,保证时间顺序
    for user_id in user_item_time_dict:
        user_item_time_dict[user_id].sort(key=lambda x: x[1])
        # sort(key=lambda x: x[1])：使用列表的sort方法排序，排序依据（key）是元组的第二个元素（即timestamp，时间戳）
        # （时间戳越小越靠前，即越早的点击排在前面），这对依赖时序的推荐模型（如序列推荐）至关重要。时间戳就是从 1970-01-01 00:00:00 到此刻的秒数
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
    item_cnt = defaultdict(int)  # 物品被点击次数  该方法常被用作计数器

    print("开始计算物品相似度矩阵...")
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 遍历用户的每个交互物品
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {}) # 等价于 if i not in i2i_sim :  i2i_sim[i] = {}
            # 遍历同一用户点击的其他物品
            for j, j_click_time in item_time_list:
                # 通过双层循环（i和j），统计同一用户点击的所有物品对(i,j)的共现情况
                if i == j:
                    # 若i和j是同一个物品（自己和自己），则跳过当前循环（无需计算物品与自身的相似度）
                    continue
                # 初始化物品j的相似度
                i2i_sim[i].setdefault(j, 0)
                # 计算相似度,考虑用户的历史序列长度作为惩罚项
                # 序列越长,物品间相似度越小
                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1) # 1 / math.log(len(item_time_list) + 1)：惩罚项 1防止0取对数 e为底数

    # 3. 对相似度进行余弦归一化
    i2i_sim_ = i2i_sim.copy() # 浅拷贝 浅拷贝拷贝地址（一动跟着动） 深拷贝拷贝内容
    print("开始相似度归一化...")
    for i, related_items in tqdm(i2i_sim.items()):
        for j, wij in related_items.items():
            # 使用余弦相似度公式: wij / sqrt(|Ui| * |Uj|)
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])  # 单向共现

    # 4. 保存相似度矩阵
    save_path = './data/'
    if not os.path.exists(save_path): # os.path.exists(save_path)：判断data文件夹是否存在
        os.makedirs(save_path) # 若不存在，则通过os.makedirs(save_path)创建该文件夹
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb')) # open(...)：以二进制写入模式（'wb'）打开文件itemcf_i2i_sim.pkl
    # 序列化（Serialization）的核心就是将数据结构或对象转化为可存储或可传输的二进制序列（字节流）
    # w表示 “写入模式”：如果文件不存在，会创建一个新文件；如果文件已存在，会覆盖原有内容。
    # b表示 “二进制模式”：与pickle的序列化格式匹配（pickle处理的是二进制数据
    return i2i_sim_


def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
    基于物品的协同过滤推荐

    Args:
        user_id: 目标用户id
        user_item_time_dict: 用户交互历史字典
        i2i_sim: 物品相似度矩阵
        sim_item_topk: 每个物品取最相似的k个物品
        recall_item_num: 最终召回的物品数量
        item_topk_click: 热门物品列表(用于补充推荐)

    Returns:
        list: 推荐物品列表 [(item_id, score), ...]

    推荐步骤:
    1. 获取用户历史交互物品
    2. 遍历历史物品,找到与其最相似的k个物品
    3. 过滤掉已交互的物品
    4. 如果不足recall_item_num,用热门物品补充
    """

    # 1. 获取用户历史交互物品
    user_hist_items = user_item_time_dict[user_id]
    # user_hist_items 是一个列表 里面的元素是元组
    user_hist_items_set = {item_id for item_id, _ in user_hist_items}
    # user_hist_items_set：提取历史物品的 ID 并转为集合（忽略时间戳），用于快速判断物品是否被用户点击过（集合的in操作效率为 O (1)）。

    # 2. 计算物品评分
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items): # 变量个数要和欲拆包的迭代对象的值个数相等才能拆包
        # 获取与物品i最相似的k个物品
        # 由于 user_hist_items 是按时间戳排过序的  所以loc具有时间意义 loc越大越近
        similar_items = sorted(i2i_sim.get(i, {}).items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]
        # i2i_sim是字典套字典   .items() 可迭代的视图对象（近似列表套元组）sorted参数可以是一切迭代对象（包括字典视图）reverse=True 降序
        for j, wij in similar_items:
            if j in user_hist_items_set:  # 过滤掉用户已交互的物品
                continue
            # 累加相似度得分
            item_rank.setdefault(j, 0)
            # 如果字典 item_rank 中 不存在键 j，就插入 j: 0 并返回 0；
            # 如果 已存在键 j，则什么也不做，直接返回该键对应的值。
            item_rank[j] += wij

    # 3. 补充热门物品
    if len(item_rank) < recall_item_num: # len()的参数范围也是一切可迭代对象
        for i, item in enumerate(item_topk_click):
            if item in item_rank:
                continue
            item_rank[item] = -i - 100  # 负数分数保证热门物品排在后面 但热门物品之间 热门的在前边
            if len(item_rank) == recall_item_num:
                break

    # 4. 排序并截断
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank


"""
主函数流程
"""
print("开始执行推荐系统主流程...")

print("\n第1步: 获取热门文章列表...")
# 1. 获取热门物品列表(点击量最高的N篇文章)
item_topk_click = Data_analysis.train_df['click_article_id'].value_counts().index.tolist()
# .value_counts()：pandas的 Series 方法，统计每个文章 ID 的点击次数(出现次数），并按点击量降序排序（默认行为）
# .index：获取排序后的文章 ID（value_counts()的结果中，索引是文章 ID，值是点击次数）。 返回的是 pandas.Index 对象
# .tolist()：将索引转换为列表，得到[文章ID1, 文章ID2, ...]（点击量从高到低排列）。
print(f"获取到{len(item_topk_click)}篇热门文章")

print("\n第2步: 计算文章相似度矩阵...")
# 2. 计算物品相似度矩阵
i2i_sim = itemcf_sim(Data_analysis.train_df)
print(f"相似度矩阵计算完成,共有{len(i2i_sim)}篇文章的相似度信息")

print("\n第3步: 构建用户交互历史字典...")
# 3. 获取测试集用户-物品-时间交互字典
user_item_time_dict = get_user_item_time(Data_analysis.test_df)
print(f"构建完成,共有{len(user_item_time_dict)}个用户的交互记录")

print("\n第4步: 准备进行推荐...")
# 4. 对测试集用户进行推荐
test_users = Data_analysis.test_df['user_id'].unique()
print(f"需要对{len(test_users)}个用户进行推荐")

# 推荐参数
sim_item_topk = 5  # 相似物品数量
recall_item_num = 5  # 召回物品数量
print(f"推荐参数设置: 每篇文章取{sim_item_topk}个相似文章,每个用户推荐{recall_item_num}篇文章")

# 存储所有用户的推荐结果
user_recall_items = {}

print("\n开始为每个用户生成推荐...")
for user_id in tqdm(test_users):
    user_recall_items[user_id] = item_based_recommend(
        user_id, user_item_time_dict, i2i_sim,
        sim_item_topk, recall_item_num, item_topk_click
    )
# item_based_recommend() 返回值是一个列表 里面的元素是元组包含物品id和相似度打分
print(f"\n推荐完成! 共生成{len(user_recall_items)}个用户的推荐结果")

# 生成提交文件
print("开始生成提交文件...")

# 创建结果列表
submit_list = []

# 从已有的推荐结果生成提交格式
for user_id in test_users:
    # 获取该用户的推荐结果
    item_list = user_recall_items[user_id]
    # 提取文章id
    article_list = [x[0] for x in item_list]
    # 如果不足5个，用最后一个补齐  重复推荐最后一个
    if len(article_list) < 5:
        article_list.extend([article_list[-1]] * (5 - len(article_list)))

    # 添加到结果列表
    submit_list.append([user_id] + article_list[:5])

# 转换为DataFrame
submit_df = pd.DataFrame(
    submit_list,
    columns=['user_id', 'article_1', 'article_2', 'article_3', 'article_4', 'article_5']
)

# 按照user_id排序
submit_df = submit_df.sort_values('user_id').reset_index(drop=True)
# sort_values('user_id')：对submit_df按照user_id列的值进行升序排序
# reset_index()：重置数据框的索引（行标签），默认会保留原来的索引作为新列 drop=True：参数表示丢弃原来的索引，不保留为新列，仅生成从 0 开始的新索引

# 保存结果
save_path = './data/submit.csv'
submit_df.to_csv(save_path, index=False)
# to_csv(save_path, ...)：pandas中 DataFrame 的方法，用于将数据框写入 CSV 文件
# save_path：字符串类型，指定 CSV 文件的保存路径（如'./submit/result.csv'），包含文件名和路径。若路径不存在，可能会报错（需提前确保目录存在，可通过os.makedirs创建）
# index=False：index默认为True，表示保存 CSV 时会将 DataFrame 的索引（行标签）作为一列写入文件  0 1 2 ...
print(f"提交文件已生成，保存至: {save_path}")
print(f"推荐结果shape: {submit_df.shape}")
print("\n前5行示例:")
print(submit_df.head())
