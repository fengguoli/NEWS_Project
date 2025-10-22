import math
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import EmbeddingBased


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
    user_hist_items_set = {item_id for item_id, _ in user_hist_items}

    # 2. 计算物品评分
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        # 获取与物品i最相似的k个物品
        similar_items = sorted(i2i_sim.get(i, {}).items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]

        for j, wij in similar_items:
            if j in user_hist_items_set:  # 过滤掉用户已交互的物品
                continue

            # 累加相似度得分
            item_rank.setdefault(j, 0)
            item_rank[j] += wij

    # 3. 补充热门物品
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank:
                continue
            item_rank[item] = -i - 100  # 负数分数保证热门物品排在后面
            if len(item_rank) == recall_item_num:
                break

    # 4. 排序并截断
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank

# 测试集划分出每个user的最后1个interaction作为test ground truth，其他作为test可见interaction
test_ground_truth = EmbeddingBased.new_test_df.sort_values(by='click_timestamp', ascending=False).groupby('user_id').tail(1)
# new_test_df.sort_values(by='click_timestamp', ascending=False)：按照时间戳降序排序，确保最新的交互排在前面
# .groupby('user_id')：按照 user_id 分组，将每个用户的数据分到不同的组中
# 对每个用户组取最后1行。由于已经按时间戳降序排序，这将提取每个用户的最后1个交互
test_remain = EmbeddingBased.new_test_df.loc[~EmbeddingBased.new_test_df.index.isin(test_ground_truth.index)]
# new_test_df.loc[...]：使用布尔索引选择 new_test_df 中不在 test_ground_truth 索引中的行

# 获取测试集用户
test_users = test_remain['user_id'].unique()
# 获取训练集热门物品
item_topk_click = EmbeddingBased.new_train_df['click_article_id'].value_counts().index.tolist()
# .index.tolist()：提取点击次数最多的物品的 ID，并将其转换为列表
# 构建测试集用户的历史交互字典
test_user_item_time_dict = {}
for user, hist in test_remain.groupby('user_id'):
    test_user_item_time_dict[user] = list(zip(hist['click_article_id'], hist['click_timestamp']))
    # zip(hist['click_article_id'], hist['click_timestamp'])： 将每个用户的点击物品 ID 和点击时间戳配对，生成一个列表

# 使用itemcf sim进行推荐
itemcf_recall_dict = {}
for user in tqdm(test_users):
    itemcf_recall_dict[user] = item_based_recommend(user, user_item_time_dict=test_user_item_time_dict,
                                                    i2i_sim=EmbeddingBased.i2i_sim, sim_item_topk=10, recall_item_num=5, item_topk_click=item_topk_click)

# 使用item embedding sim进行推荐
embedding_recall_dict = {}
for user in tqdm(test_users):
    embedding_recall_dict[user] = item_based_recommend(user, user_item_time_dict=test_user_item_time_dict,
                                                       i2i_sim=EmbeddingBased.i2i_embsim, sim_item_topk=10, recall_item_num=5, item_topk_click=item_topk_click)


# Hit Rate@k 评估指标:
# Hit Rate@k 表示在推荐的前k个物品中,有多少用户的实际点击物品被包含在内
# 计算方式是: 推荐列表命中用户实际点击物品的用户数 / 总用户数
# 该指标用于评估推荐系统的召回效果,越高说明推荐结果与用户实际兴趣越相关

def get_hit_rate(recall_dict, ground_truth):
    """计算hit rate @ k"""
    hit_count = 0
    total_user = len(recall_dict)
    for user, recall_items in recall_dict.items():
        gt_item = ground_truth[user]
        for item, score in recall_items:
            if item == gt_item:
                hit_count += 1
                break

    return hit_count * 1.0 / total_user


ground_truth_label = test_ground_truth[['user_id', 'click_article_id']].set_index('user_id').to_dict()[
    'click_article_id']
# .set_index('user_id')：将 user_id 列设置为索引。这样，每一行的 user_id 成为索引，click_article_id 成为对应的值
# .to_dict()将 DataFrame 转换为字典 默认情况下，to_dict() 会将每一列转换为一个字典，其中索引作为键，列值作为值。
# ['click_article_id']：从生成的字典中提取 click_article_id 列对应的字典
# 计算itemcf的hit rate
itemcf_hit_rate = get_hit_rate(itemcf_recall_dict, ground_truth_label)
print("ItemCF的命中率: ", round(itemcf_hit_rate, 4))

# 计算item embedding的hit rate
embedding_hit_rate = get_hit_rate(embedding_recall_dict, ground_truth_label)
print("Item Embedding的命中率: ", round(embedding_hit_rate, 4))

# 将两路召回结合使用
class HybridRecommender:
    def __init__(self,
                 i2i_sim,  # ItemCF相似度矩阵
                 i2i_embsim,  # Embedding相似度矩阵
                 train_df,  # 训练数据，用于计算物品统计信息
                 cf_weight=0.7,
                 emb_weight=0.3,
                 sim_item_topk=20,
                 cf_item_topk=20,
                 emb_item_topk=20,
                 recall_item_num=10,
                 alpha=0.5,  # 调节时间衰减的参数
                 min_interaction_threshold=5  # 最小交互阈值
                 ):
        """
        混合推荐器初始化

        参数:
        - i2i_sim: ItemCF相似度矩阵
        - i2i_embsim: Embedding相似度矩阵
        - train_df: 训练数据DataFrame
        - cf_weight: ItemCF的基础权重
        - emb_weight: Embedding的基础权重
        - sim_item_topk: 每个物品取最相似的k个物品
        - cf_item_topk: ItemCF召回路的物品数量
        - emb_item_topk: Embedding召回路的物品数量
        - recall_item_num: 最终召回的物品数量
        - alpha: 时间衰减因子
        - min_interaction_threshold: 最小交互阈值，用于判断冷启动物品
        """
        self.i2i_sim = i2i_sim
        self.i2i_embsim = i2i_embsim
        self.cf_weight = cf_weight
        self.emb_weight = emb_weight
        self.sim_item_topk = sim_item_topk
        self.cf_item_topk = cf_item_topk
        self.emb_item_topk = emb_item_topk
        self.recall_item_num = recall_item_num
        self.alpha = alpha
        self.min_interaction_threshold = min_interaction_threshold

        # 计算物品统计信息
        self.calculate_item_stats(train_df)

        # 获取热门物品列表
        self.item_topk_click = train_df['click_article_id'].value_counts().index.tolist()

    def calculate_item_stats(self, df):
        """计算物品统计信息"""
        # 计算物品交互次数
        self.item_interaction_count = df['click_article_id'].value_counts().to_dict()

        # 计算物品的平均交互时间
        item_timestamps = df.groupby('click_article_id')['click_timestamp'].mean()
        self.item_mean_timestamp = item_timestamps.to_dict()

    def get_dynamic_weights(self, item_id):
        """
        根据物品的交互情况动态计算权重

        """
        interaction_count = self.item_interaction_count.get(item_id, 0)
        # 根据交互数量动态调整权重
        cf_w = min(0.95, self.cf_weight + 0.1 * math.log(1 + interaction_count))
        emb_w = 1 - cf_w

        return cf_w, emb_w

    def get_time_decay_factor(self, item_id):
        """计算时间衰减因子"""
        mean_time = self.item_mean_timestamp.get(item_id, self.current_timestamp)
        time_diff = (self.current_timestamp - mean_time) / (24 * 60 * 60 * 1000)  # 转换为天数
        return math.exp(-self.alpha * time_diff)

    def hybrid_recommend(self, user_id, user_item_time_dict, current_timestamp):
        """
        混合推荐方法

        参数:
        - user_id: 用户ID
        - user_item_time_dict: 用户交互历史字典

        返回:
        - 推荐物品列表 [(item_id, score), ...]
        """
        # 1. 获取用户历史交互
        user_hist_items = user_item_time_dict.get(user_id, [])
        user_hist_items_set = {item_id for item_id, _ in user_hist_items}

        # 2. 分别计算两种相似度的物品评分
        item_rank_cf = defaultdict(float)
        item_rank_emb = defaultdict(float)

        # 更新当前时间戳
        self.current_timestamp = current_timestamp

        # 遍历历史交互物品
        for loc, (i, click_time) in enumerate(user_hist_items):
            # 获取ItemCF相似物品
            cf_similar_items = sorted(self.i2i_sim.get(i, {}).items(),
                                      key=lambda x: x[1],
                                      reverse=True)[:self.sim_item_topk]
            # 获取Embedding相似物品
            emb_similar_items = sorted(self.i2i_embsim.get(i, {}).items(),
                                       key=lambda x: x[1],
                                       reverse=True)[:self.sim_item_topk]

            # 计算时间衰减权重
            time_weight = 1.0 / (1.0 + 0.1 * (len(user_hist_items) - loc))

            # 累积ItemCF分数
            for j, cf_sim in cf_similar_items:
                if j in user_hist_items_set:
                    continue
                item_rank_cf[j] += cf_sim * time_weight

            # 累积Embedding分数
            for j, emb_sim in emb_similar_items:
                if j in user_hist_items_set:
                    continue
                item_rank_emb[j] += emb_sim * time_weight

        # item_rank_cf 和 item_rank_emb 中只保留前cf_topk和emb_topk个物品
        item_rank_cf = dict(sorted(item_rank_cf.items(), key=lambda x: x[1], reverse=True)[:self.cf_item_topk])
        item_rank_emb = dict(sorted(item_rank_emb.items(), key=lambda x: x[1], reverse=True)[:self.emb_item_topk])

        # 3. 融合两种相似度的结果
        item_rank = defaultdict(float)
        all_items = set(item_rank_cf.keys()) | set(item_rank_emb.keys())

        for item in all_items:
            # 获取动态权重
            cf_w, emb_w = self.get_dynamic_weights(item)

            # 计算时间衰减因子
            time_decay = self.get_time_decay_factor(item)

            # 融合得分
            cf_score = item_rank_cf.get(item, 0)
            emb_score = item_rank_emb.get(item, 0)

            # # 归一化处理
            # if cf_score > 0:
            #     cf_score = cf_score / max(item_rank_cf.values()) if item_rank_cf else 0
            # if emb_score > 0:
            #     emb_score = emb_score / max(item_rank_emb.values()) if item_rank_emb else 0

            # 最终得分
            item_rank[item] = (cf_w * cf_score + emb_w * emb_score) * time_decay
        # 4. 补充热门物品
        if len(item_rank) < self.recall_item_num:
            for i, item in enumerate(self.item_topk_click):
                if item in item_rank or item in user_hist_items_set:
                    continue
                item_rank[item] = -i - 100  # 负数分数保证热门物品排在后面
                if len(item_rank) == self.recall_item_num:
                    break

        # 5. 排序并返回
        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)

        return item_rank[:self.recall_item_num]

    def evaluate_recommendation(self, test_users, test_user_time, test_user_item_time_dict, test_ground_truth):
        """
        评估推荐结果

        参数:
        - test_users: 测试集用户列表
        - test_user_item_time_dict: 测试集用户交互历史
        - test_ground_truth: 测试集真实点击数据

        返回:
        - hit_rate: 命中率
        - diversity: 多样性（不同分类的比例）
        """
        hit_count = 0
        rec_items_total = set()

        for user, current_timestamp in tqdm(zip(test_users, test_user_time)):
            # 获取推荐结果
            rec_items = self.hybrid_recommend(user, test_user_item_time_dict, current_timestamp)
            # 记录所有推荐的物品
            rec_items_total.update([item for item, _ in rec_items])

            # 计算命中
            gt_item = test_ground_truth[user]
            for rec_item, _ in rec_items:
                if gt_item == rec_item:
                    hit_count += 1
                    break

        # 计算指标
        hit_rate = hit_count / len(test_users)
        diversity = len(rec_items_total) / (len(test_users) * self.recall_item_num)

        return hit_rate, diversity

# 分别计算i2i_sim和i2i_embsim的相似度均值
i2i_sim_mean = 0
for i in EmbeddingBased.i2i_sim.keys():
    i2i_sim_mean += np.array(list(EmbeddingBased.i2i_sim[i].values())).mean()
i2i_sim_mean = i2i_sim_mean / len(EmbeddingBased.i2i_sim)
i2i_embsim_mean = 0
for i in EmbeddingBased.i2i_embsim.keys():
    i2i_embsim_mean += np.array(list(EmbeddingBased.i2i_embsim[i].values())).mean()
i2i_embsim_mean = i2i_embsim_mean / len(EmbeddingBased.i2i_embsim)
print(i2i_sim_mean,i2i_embsim_mean)

# 根据该均值调整cf和emb的weight
cf_weight = i2i_embsim_mean / (i2i_sim_mean + i2i_embsim_mean)
emb_weight = i2i_sim_mean / (i2i_sim_mean + i2i_embsim_mean)
print(cf_weight,emb_weight)

# 初始化混合推荐器
recommender = HybridRecommender(
    i2i_sim=EmbeddingBased.i2i_sim,
    i2i_embsim=EmbeddingBased.i2i_embsim,
    train_df=EmbeddingBased.new_train_df,
    cf_weight=cf_weight,
    emb_weight=emb_weight,
    sim_item_topk=20,
    cf_item_topk=20,
    emb_item_topk=5, # 由于embedding召回效果较差，所以只取5个
    recall_item_num=5,
    alpha=0.5,
    min_interaction_threshold=5
)

# 对测试集用户进行推荐并评估
test_users, test_user_time = test_ground_truth['user_id'].to_list(), test_ground_truth['click_timestamp'].to_list()
hit_rate, diversity = recommender.evaluate_recommendation(
    test_users=test_users[:1000], test_user_time=test_user_time[:1000],
    test_user_item_time_dict=test_user_item_time_dict,
    test_ground_truth=ground_truth_label
)

print(f"混合推荐的命中率（HR@5）: {hit_rate:.4f}")

print(f"推荐结果的多样性: {diversity:.4f}")
