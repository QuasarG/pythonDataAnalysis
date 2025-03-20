"""
week3.csv 包含广州八大热门糖水店的评论数据
包括顾客ID、评论时间、评分、评论内容以及口味、环境、服务、店铺ID等信息（已分词，空格分割）
五类情绪字典emotion_lexicon，包括anger、disgust、fear、sadness和joy
"""
import pandas as pd
import random
import matplotlib.pyplot as plt

# 使用闭包实现情绪词典的惰性加载和持久化
def load_emotion_dict():
    # 闭包中的持久化变量
    emotion_dict = {}
    emotion_types = ['anger', 'disgust', 'fear', 'sadness', 'joy']
    is_loaded = False
    
    def get_emotion_dict():
        nonlocal emotion_dict, is_loaded
        # 惰性加载：只在第一次调用时加载情绪词典
        if not is_loaded:
            for emotion in emotion_types:
                with open(f'emotion_lexicon/{emotion}.txt', 'r', encoding='utf-8') as f:
                    emotion_dict[emotion] = set(f.read().splitlines())
            is_loaded = True
        return emotion_dict, emotion_types
    
    return get_emotion_dict

# 创建情绪词典加载器实例
get_emotion_data = load_emotion_dict()

# 1. 情绪分析函数实现
def mix_emotion(comments_df):
    """
    函数一：
    混合情绪分析：认为文本的情绪是混合的，即统计文本中所有情绪词的出现次数，
    并计算每种情绪的比例。例如，如果joy出现n1次，总情绪词数为n，则joy的比例为n1/n。
    """
    # 获取情绪词典和情绪类型
    emotion_dict, emotion_types = get_emotion_data()
    
    # 初始化结果DataFrame，包含每种情绪的比例
    result_df = pd.DataFrame(columns=emotion_types)
    
    # 遍历每条评论，分析每一行的情绪
    for index, row in comments_df.iterrows():
        # 检查评论内容是否为NaN
        if pd.isna(row['cus_comment']):
            # 如果是NaN，则设置所有情绪比例为0
            result_df.loc[index] = {emo: 0 for emo in emotion_types}
            continue
            
        # 获取评论内容并按空格分词
        words = row['cus_comment'].split()
        
        # 统计每种情绪词出现的次数
        count_emotion = {emo: 0 for emo in emotion_types}
        for word in words:
            for emotion in emotion_types:
                if word in emotion_dict[emotion]:
                    count_emotion[emotion] += 1
        
        # 计算总情绪词数
        total_emotion = sum(count_emotion.values())
        
        # 计算每种情绪的比例
        if total_emotion > 0:
            emotion_ratio = {emo: count/total_emotion for emo, count in count_emotion.items()}
        else:
            emotion_ratio = {emo: 0 for emo in emotion_types}
        
        # 将结果添加到DataFrame中
        result_df.loc[index] = emotion_ratio
    
    return result_df
        
def single_emotion(comments_df):
    """
    函数二：
    唯一情绪分析：认为一条文本的情绪是唯一的，即根据情绪词出现次数最多的情绪来确定文本的整体情绪。
    例如，如果anger的情绪词出现次数最多，则该评论的情绪应标记为angry。
    """
    # 获取情绪词典和情绪类型
    emotion_dict, emotion_types = get_emotion_data()
    
    # 初始化结果Series，用于存储每条评论的主要情绪
    result_series = pd.Series(index=comments_df.index)
    
    # 遍历每条评论，分析每一行的情绪
    for index, row in comments_df.iterrows():
        # 检查评论内容是否为NaN
        if pd.isna(row['cus_comment']):
            # 如果是NaN，则标记为'neutral'
            result_series[index] = 'neutral'
            continue
            
        # 获取评论内容并按空格分词
        words = row['cus_comment'].split()
        
        # 统计每种情绪词出现的次数
        count_emotion = {emo: 0 for emo in emotion_types}
        for word in words:
            for emotion in emotion_types:
                if word in emotion_dict[emotion]:
                    count_emotion[emotion] += 1
        
        # 找出出现次数最多的情绪
        if sum(count_emotion.values()) > 0:
            # 找到最大值
            max_count = max(count_emotion.values())
            # 找出所有具有最大值的情绪
            max_emotions = [emo for emo, count in count_emotion.items() if count == max_count]
            
            # 如果有多个情绪词出现次数相同且最多，随机选择一个
            if len(max_emotions) > 1:
                result_series[index] = random.choice(max_emotions)
            else:
                result_series[index] = max_emotions[0]
        else:
            # 如果没有情绪词出现，标记为'neutral'
            result_series[index] = 'neutral'
    
    # 将情绪结果添加到原始DataFrame中的新列'emotion'
    comments_df['emotion'] = result_series
    
    return result_series

# 2. 时间模式分析函数实现
def time_analysis(mode: int, comments_df, emotype: int, shop_id: str):
    """
    利用数据集中的评论时间信息，分析不同时间段的情绪比例变化趋势。
    实现一个函数，可以通过参数控制来返回指定店铺、指定情绪的时间模式，
    并可视化呈现这些模式。例如，可以展示shopID为518986的店铺积极情绪的小时模式，
    或shopID为520004店铺消极情绪的周模式等。
    """
    # mode代表模式，小时模式（0） or 周模式（1）
    # emotype代表选择消极（0） or 积极（1）
    # 获取shop_id为xxxxx的商户消极/积极情绪的模式
    
    # 获取情绪词典和情绪类型
    emotion_dict, emotion_types = get_emotion_data()

    comments_df['comment_time'] = pd.to_datetime(comments_df['comment_time'])
    
    # 提取星期信息（0=周一，6=周日）
    comments_df['weekday'] = comments_df['comment_time'].dt.weekday
    # 提取小时信息
    comments_df['hour'] = comments_df['comment_time'].dt.hour

    # 筛选指定店铺
    # 确保shop_id的类型与数据集中的shopID类型一致
    # 打印调试信息
    print(f"筛选店铺ID: {shop_id}, 类型: {type(shop_id)}")
    print(f"数据集中shopID的类型: {comments_df['shopID'].dtype}")
    print(f"数据集中shopID的唯一值: {comments_df['shopID'].unique()[:5]}...")
    
    # 尝试将shop_id转换为与数据集中相同的类型
    if comments_df['shopID'].dtype == 'int64':
        shop_id = int(shop_id)
    
    # 筛选指定店铺
    shop_data = comments_df[comments_df['shopID'] == shop_id]
    print(f"筛选后的店铺数据数量: {len(shop_data)}")
    
    # 如果筛选结果为空，尝试不同的筛选方法
    if len(shop_data) == 0:
        print("尝试使用字符串比较筛选...")
        shop_data = comments_df[comments_df['shopID'].astype(str) == str(shop_id)]
        print(f"字符串比较筛选后的数据数量: {len(shop_data)}")

    # 检查emotion列是否存在
    if 'emotion' not in comments_df.columns:
        print("警告: 数据集中不存在emotion列，请先运行single_emotion函数")
        # 返回空的Series
        if mode == 0:
            return pd.Series()
        else:
            return pd.Series()

    # 筛选指定情绪
    if emotype == 0:  # 消极情绪
        emotion_data = shop_data[shop_data['emotion'].isin(['anger', 'disgust', 'fear', 'sadness'])]
        # 如果筛选结果为空，打印调试信息
        if emotion_data.empty:
            print(f"警告: 店铺{shop_id}没有消极情绪数据")
            print(f"该店铺情绪分布: {shop_data['emotion'].value_counts()}")
    else:  # 积极情绪
        emotion_data = shop_data[shop_data['emotion'] == 'joy']
        # 如果筛选结果为空，打印调试信息
        if emotion_data.empty:
            print(f"警告: 店铺{shop_id}没有积极情绪数据")
            print(f"该店铺情绪分布: {shop_data['emotion'].value_counts()}")
    
    # 保存筛选后的数据到CSV文件
    csv_file_name = f'shop_{shop_id}_{"positive" if emotype == 1 else "negative"}_{"hourly" if mode == 0 else "weekly"}_data.csv'
    emotion_data.to_csv(csv_file_name, encoding='utf-8', index=False)
    print(f"筛选后的数据已保存到 {csv_file_name}")
    
    # 按小时分组统计
    hourly_counts = emotion_data.groupby('hour').size()

    # 按星期分组统计
    weekly_counts = emotion_data.groupby('weekday').size()

    # 接下来进行可视化
    # 解决画图字体问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体

    # 根据模式和情绪类型生成不同的图片标题和文件名
    mode_text = '小时' if mode == 0 else '星期'
    emotion_text = '积极' if emotype == 1 else '消极'
    
    if mode == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(hourly_counts.index, hourly_counts.values, marker='o')
        plt.xticks(range(0,24,1))  # 设置每小时一个刻度
        plt.title(f'Shop{shop_id}{emotion_text}情绪{mode_text}分布')
        plt.xlabel('小时')
        plt.ylabel('评论数量')
        plt.gcf().autofmt_xdate()  # 自动旋转日期标签
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(weekly_counts.index, weekly_counts.values, marker='o')
        plt.title(f'Shop{shop_id}{emotion_text}情绪{mode_text}分布')
        plt.xlabel('星期')
        plt.ylabel('评论数量')
    
    # 生成唯一的文件名，避免覆盖
    file_name = f'shop_{shop_id}_{"positive" if emotype == 1 else "negative"}_{"hourly" if mode == 0 else "weekly"}.png'
    plt.savefig(file_name)

    if mode == 0:
        return hourly_counts
    else:
        return weekly_counts

# 3. 情绪分析与评分对比
def judge_emotion(comments_df):
    """
    将情绪分析的结果与数据集中的产品的评分进行对比分析，以合适的方式进行可视化呈现，
    并借此分析情绪词典分析的缺点（如无法覆盖的比例等）。同时，还可以借助数据集中的其他指标，
    开拓分析思路。比如，带有积极情绪的差评是否对其他顾客或店铺是不是更有用（usefulness)?
    """
    # 分析情绪与评分的关系
    # 计算每种情绪的平均评分，使用stars列
    emotion_score = comments_df.groupby('emotion')['stars'].mean()
    
    # 计算情绪词典的覆盖率
    total_comments = len(comments_df)
    neutral_comments = len(comments_df[comments_df['emotion'] == 'neutral'])
    coverage_rate = (total_comments - neutral_comments) / total_comments * 100
    
    # 分析积极情绪的差评
    # 定义差评（小于等于3分）和好评（大于3分）
    low_score_joy = comments_df[(comments_df['emotion'] == 'joy') & (comments_df['stars'] <= 3)]
    high_score_joy = comments_df[(comments_df['emotion'] == 'joy') & (comments_df['stars'] > 3)]
    
    # 可视化
    # 解决画图字体问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.figure(figsize=(15, 5))
    
    # 情绪评分对比
    plt.subplot(131)
    
    # 将joy放在最后
    emotion_order = [e for e in emotion_score.index if e != 'joy']
    if 'joy' in emotion_score.index:
        emotion_order.append('joy')
    # 创建颜色列表，joy使用不同颜色
    colors = ['#1f77b4'] * len(emotion_order)
    if 'joy' in emotion_score.index:
        colors[-1] = '#ff7f0e'  # 为joy设置不同颜色
    
    # 绘制条形图
    bars = plt.bar(emotion_order, [emotion_score[e] for e in emotion_order], color=colors)
    
    # 在条形上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('不同情绪的平均评分')
    plt.xlabel('情绪类型')
    plt.ylabel('平均评分')
    plt.xticks(rotation=45)
    
    # 情绪词典覆盖率
    plt.subplot(132)
    plt.pie([neutral_comments, total_comments - neutral_comments],
    labels=['未覆盖', '已覆盖'],
    autopct='%1.1f%%')
    plt.title('情绪词典覆盖率')
    
    # 积极情绪评分分布
    plt.subplot(133)
    bars = plt.bar(['差评', '好评'],[len(low_score_joy), len(high_score_joy)])
    
    # 在条形上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height}', ha='center', va='bottom')
                
    plt.title('积极情绪的评分分布')
    plt.ylabel('评论数量')
    
    plt.tight_layout()
    plt.savefig('emotion_analysis.png')
    
    # 返回分析结果
    analysis_result = {
        'emotion_score': emotion_score,
        'coverage_rate': coverage_rate,
        'positive_negative_ratio': len(low_score_joy) / len(high_score_joy) if len(high_score_joy) > 0 else 0
    }
    
    return analysis_result

def main():
    # 打开文件并读取评论数据
    comments_df = pd.read_csv('week3.csv', encoding='utf-8')
    # 创建情绪词典加载器
    global get_emotion_data
    get_emotion_data = load_emotion_dict()

    # 1. 情绪分析实现
    mix_result = mix_emotion(comments_df)
    single_result = single_emotion(comments_df)
    
    # 保存结果到CSV文件
    mix_result.to_csv('mix_emotion_result.csv')
    pd.DataFrame({'emotion': single_result}).to_csv('single_emotion_result.csv')
    
    # 2. 时间模式分析实现
    # 选择一个代表性的店铺ID进行分析
    shop_id = '518986'
    
    # 小时模式-积极情绪
    hourly_positive = time_analysis(0, comments_df, 1, shop_id)
    print(f"\n店铺{shop_id}积极情绪的小时分布已保存为 shop_{shop_id}_positive_hourly.png")
    
    # 小时模式-消极情绪
    hourly_negative = time_analysis(0, comments_df, 0, shop_id)
    print(f"店铺{shop_id}消极情绪的小时分布已保存为 shop_{shop_id}_negative_hourly.png")
    
    # 周模式-积极情绪
    weekly_positive = time_analysis(1, comments_df, 1, shop_id)
    print(f"店铺{shop_id}积极情绪的周分布已保存为 shop_{shop_id}_positive_weekly.png")
    
    # 周模式-消极情绪
    weekly_negative = time_analysis(1, comments_df, 0, shop_id)
    print(f"店铺{shop_id}消极情绪的周分布已保存为 shop_{shop_id}_negative_weekly.png")

    # 3. 情绪分析与评分对比
    analysis_result = judge_emotion(comments_df)
    print("\n情绪分析结果：")
    print(f"情绪词典覆盖率: {analysis_result['coverage_rate']:.2f}%")
    print(f"积极情绪差评与好评比例: {analysis_result['positive_negative_ratio']:.4f}")
    print("各情绪平均评分:")
    print(analysis_result['emotion_score'])
    print("\n情绪分析图表已保存为 emotion_analysis.png")
    print("\n混合情绪分析结果已保存为 mix_emotion_result.csv")
    print("单一情绪分析结果已保存为 single_emotion_result.csv")


if __name__ == "__main__":
    main()