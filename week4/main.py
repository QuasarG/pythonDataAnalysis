import pandas as pd
import matplotlib.pyplot as plt
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class LDA_model:
    def __init__(self):
        pass
    # 1. 文档预处理
    # 实现一个模块，其中包含一个或多个函数，其能够读取该数据集并将之分店铺（共8家店铺，可根据shopID进行区分）
    # 将评论合并为以天为单位的文档集合
    
    # 1.1 读取数据集
    def read_csv(self, path):
        df = pd.read_csv(path, encoding='utf-8')
        return df
    
    # 1.2 将某一店铺的评论按天进行合并
    def merge_by_day(self, df):
        # 从comment_time提取日期
        df['date'] = pd.to_datetime(df['comment_time']).dt.date
        # 数据清洗：将非字符串类型的评论转换为空字符串
        df['cus_comment'] = df['cus_comment'].fillna('').astype(str)
        # 按shopID和date进行分组，并将评论合并为一个文档
        grouped = df.groupby(['shopID', 'date'])['cus_comment'].apply(lambda x: ' '.join(x))
        return grouped # 返回一个Series，索引为shopID和date，值为合并后的评论
    
    # 2. 文本的特征表示
    # 通过一个或多个函数，将每个文档转变为词频特征表示，以形成文档-词语的词频矩阵，
    # 可以选择使用sklearn中的CountVectorizer和TfidfVectorizer两种方式

    # 2.1 将每个文档转变为词频特征表示，形成文档-词语的词频矩阵
    def count_vectorizer(self, grouped):
        # 导入CountVectorizer
        from sklearn.feature_extraction.text import CountVectorizer
        # 实例化CountVectorizer
        self.vectorizer = CountVectorizer()
        # 对grouped进行词频统计
        X = self.vectorizer.fit_transform(grouped)
        return X, self.vectorizer

    # 3. 文本的话题分析
    # 通过一个或多个函数，借助sklearn.decomposition中的LatentDirichletAllocation构建主题模型（话题数目可以自主指定）
    
    # 3.1 借助sklearn.decomposition中的LatentDirichletAllocation构建主题模型
    # 最佳主题数目可以通过困惑度来确定
    def lda(self, X, n_topics=None):
        # 导入LatentDirichletAllocation
        from sklearn.decomposition import LatentDirichletAllocation
        import seaborn as sns
        from wordcloud import WordCloud
        import numpy as np
        
        print("开始LDA模型训练...")  # 调试信息
        
        if n_topics is None:
            print("自动选择最佳主题数...")  # 调试信息
            # 计算不同主题数下的困惑度
            perplexities = []
            topic_range = range(1, 11)
            for n in topic_range:
                lda = LatentDirichletAllocation(n_components=n, random_state=0)
                lda.fit(X)
                perplexity = lda.perplexity(X)
                perplexities.append(perplexity)
                print(f"主题数={n}，困惑度={perplexity:.2f}")  # 打印每个主题数的困惑度

            # 绘制困惑度-主题数图形
            plt.figure(figsize=(8, 5))
            plt.plot(topic_range, perplexities, 'bo-')
            plt.xlabel('Number of Topics')
            plt.ylabel('Perplexity')
            plt.title('Perplexity vs Number of Topics')
            plt.savefig('perplexity_plot.png')  # 保存图像
            plt.show()
            
            # slopes = [abs(perplexities[i] - perplexities[i-1]) for i in range(1, len(perplexities))]
            # max_slope_index = slopes.index(max(slopes)) + 1
            # n_topics = topic_range[max_slope_index]
            # print(f'最佳主题数为: {n_topics}')
            # 观察图像可得不是一直单调递减，肘部法则不适用，遂放弃，采取选择困惑度最小的主题数

            # 选择困惑度最小的主题数
            min_perplexity_index = perplexities.index(min(perplexities))
            n_topics = topic_range[min_perplexity_index]
            print(f'最佳主题数为: {n_topics} (困惑度最小)')

        print(f"使用{n_topics}个主题训练最终LDA模型...")  # 调试信息
        # 实例化LatentDirichletAllocation
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
        # 对X进行LDA
        lda.fit(X)

        # 输出每个主题对应的词语和每篇文档的主题概率分布
        print('每个主题对应的词语权重维度:', lda.components_.shape)
        print('每篇文档的主题概率分布维度:', lda.transform(X).shape)

        # 可视化每个主题的词云图
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            # 创建词云图
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                font_path='simhei.ttf')  # 设置中文字体
            # 将主题词权重转换为字典格式
            word_weights = {feature_names[i]: topic[i] for i in range(len(feature_names))}
            wordcloud.generate_from_frequencies(word_weights)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {topic_idx + 1} Word Cloud')
            plt.savefig(f'topic_{topic_idx + 1}_wordcloud.png')
            plt.show()

        # 可视化主题权重分布
        topic_weights = lda.transform(X)
        avg_topic_weights = topic_weights.mean(axis=0)
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, n_topics + 1), avg_topic_weights)
        plt.xlabel('Topic')
        plt.ylabel('Average Weight')
        plt.title('Topic Weight Distribution')
        plt.savefig('topic_weights.png')
        plt.show()

        # 可视化主题-文档分布热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(topic_weights[:20], # 只显示前20个文档
                    cmap='YlOrRd',
                    xticklabels=[f'Topic {i+1}' for i in range(n_topics)],
                    yticklabels=[f'Doc {i+1}' for i in range(20)])
        plt.title('Topic-Document Distribution (First 20 Documents)')
        plt.savefig('topic_document_heatmap.png')
        plt.show()

        return lda
    
    # 4. 序列化保存
    # 4.1 利用pickle或json对所得到的lda模型、对应的词频矩阵、以及特征表示等进行序列化保存
    def save_model(self, lda, X, vectorizer, path):
        import json
        import numpy as np
        
        # 获取每个主题的前20个最重要的词语及其权重
        n_top_words = 20
        feature_names = vectorizer.get_feature_names_out()
        top_words = []
        for topic_idx, topic in enumerate(lda.components_):
            top_word_indices = topic.argsort()[:-n_top_words-1:-1]
            words_weights = {
                feature_names[i]: float(topic[i])
                for i in top_word_indices
            }
            top_words.append(words_weights)
        
        # 将模型参数转换为可JSON序列化的格式
        model_params = {
            'n_components': lda.n_components,
            'doc_topic_prior': float(lda.doc_topic_prior) if lda.doc_topic_prior is not None else None,
            'topic_word_prior': float(lda.topic_word_prior) if lda.topic_word_prior is not None else None,
            'top_words': top_words
        }
        
        # 保存为JSON文件
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_params, f, ensure_ascii=False, indent=2)
    
    def load_model(self, path):
        import json
        import numpy as np
        from sklearn.decomposition import LatentDirichletAllocation
        from scipy.sparse import csr_matrix
        
        # 从JSON文件加载数据
        with open(path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        # 重建LDA模型
        lda = LatentDirichletAllocation(
            n_components=save_data['model_params']['n_components'],
            doc_topic_prior=save_data['model_params']['doc_topic_prior'],
            topic_word_prior=save_data['model_params']['topic_word_prior']
        )
        lda.components_ = np.array(save_data['model_params']['components'])
        lda.n_features_in_ = save_data['model_params']['n_features']
        
        # 重建词频矩阵
        X = csr_matrix(np.array(save_data['X_data']['data']))
        
        return lda, X, save_data['feature_names']
    
    # 6. 话题分布的时间趋势分析
    # 根据评论文档的时间信息，观察特定话题随着时间的变化趋势，
    # 即分析某一话题在不同时间段内的出现频率或权重，
    # 以便了解该话题在不同时期内的热度变化
    def time_trend_analysis(self, lda, X, grouped, topic_idx):
        # 数据验证
        if topic_idx >= lda.n_components:
            raise ValueError(f"话题索引{topic_idx}超出范围，最大值应为{lda.n_components-1}")
        if len(grouped) != X.shape[0]:
            raise ValueError("文档数量与特征矩阵行数不匹配")
            
        # 提取并排序日期
        dates = pd.to_datetime(grouped.index.get_level_values('date'))
        # 将日期和主题权重组合成DataFrame以便排序
        topic_weights = lda.transform(X)[:, topic_idx]
        trend_df = pd.DataFrame({'date': dates, 'weight': topic_weights})
        trend_df = trend_df.sort_values('date')
        
        # 将日期设置为索引并按周重采样计算平均值
        trend_df.set_index('date', inplace=True)
        weekly_weights = trend_df.resample('W')['weight'].mean()
        
        # 绘制趋势图
        plt.figure(figsize=(15, 8))
        plt.plot(weekly_weights.index, weekly_weights.values, 
                marker='o', linestyle='-', linewidth=1.5, markersize=3)
        
        # 设置x轴日期格式
        plt.gcf().autofmt_xdate()  # 自动旋转日期标签
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('话题权重', fontsize=12)
        plt.title(f'话题{topic_idx + 1}按周聚合的时间趋势', fontsize=14)
        
        # 优化网格显示
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend([f'话题{topic_idx + 1}的周平均权重'])
        
        # 保存图片
        plt.tight_layout()  # 自动调整布局
        plt.savefig(f'topic_{topic_idx + 1}_weight_trend.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return weekly_weights  # 返回处理后的数据供进一步分析

def main():
    lda_model = LDA_model()
    df = lda_model.read_csv('week4.csv')
    
    # 调试信息：检查数据是否加载成功
    print("数据前5行：")
    print(df.head())
    print("数据总行数：", len(df))
    
    # 对某一店铺的评论按天进行合并
    grouped = lda_model.merge_by_day(df)
    
    # 调试信息：检查合并后的文档
    print("\n合并后的文档示例：")
    print(grouped.head())
    print("合并后的文档总数：", len(grouped))
    
    # 将每个文档转变为词频特征表示，形成文档-词语的词频矩阵
    X, vectorizer = lda_model.count_vectorizer(grouped)
    
    # 调试信息：检查词频矩阵
    print("\n词频矩阵形状：", X.shape)
    print("前10个词汇：", vectorizer.get_feature_names_out()[:10])
    
    # 执行LDA模型训练
    lda = lda_model.lda(X, 3)  # 指定主题数为3，由图像可知此时困惑度最低
    
    # 保存模型、词频矩阵和特征表示
    print("\n保存模型到model.json...")
    lda_model.save_model(lda, X, vectorizer, 'model.json')
    
    # 分析每个话题的时间趋势
    for topic_idx in range(3):
        print(f"\n分析话题{topic_idx + 1}的时间趋势...")
        daily_weights = lda_model.time_trend_analysis(lda, X, grouped, topic_idx)
        print(f"话题{topic_idx + 1}的日均权重统计：")
        print(f"最大值：{daily_weights.max():.4f}")
        print(f"最小值：{daily_weights.min():.4f}")
        print(f"平均值：{daily_weights.mean():.4f}")

if __name__ == "__main__":
    main()