''' 写成函数形式，方便调用 '''
import jieba
import wordcloud
import jieba.posseg as pseg
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取文件，打印出前10行
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        return [line.strip() for line in lines]
    print("前10行文本：")
    for line in lines[:10]:
        print(line)

# 2. 分词并统计词频
def calculate_word_frequency(lines, stopwords):
    word_freq = {}
    for line in lines:
        words = jieba.lcut(line)
        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq

# 3. 按词频排序并打印前10个词
def print_top_words(word_freq):
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    print("\n词频最高的10个词：")
    for word, freq in sorted_word_freq[:10]:
        print(f"{word}: {freq}")

# 4. 引入停用词表
# 停用词来源：https://github.com/elephantnose/characters/blob/master/stop_words 作者：@elephantnose
# 自己填了几个词，忘了记录 Todo
def load_stopwords(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        stopwords = set(word.strip() for word in f.readlines())
    stopwords.update(" ")
    return stopwords

# 5. 生成词云
def generate_wordcloud(word_freq, font_path, output_path):
    wc = wordcloud.WordCloud(font_path=font_path, background_color="white")
    wc.generate_from_frequencies(word_freq)
    wc.to_file(output_path)
    print(f"词云已生成：{output_path}")

# 6. 词性分析
def analyze_pos(lines, stopwords: set):
    pos_freq = {} # 词性频率统计，格式为{词性: 频率}
    pos_words = {} # 词性对应的词，格式为{词性: {词: 频率}}
    
    for line in lines:
        words = pseg.cut(line) # 分词并标注词性
        for word, flag in words:
            if word in stopwords: # 屏蔽停用词
                continue

            pos_freq[flag] = pos_freq.get(flag, 0) + 1
            if flag not in pos_words:
                pos_words[flag] = {}
            pos_words[flag][word] = pos_words[flag].get(word, 0) + 1
    
    return pos_freq, pos_words

# 6. 打印词性频率统计
def print_top_pos(pos_freq):
    sorted_pos_freq = sorted(pos_freq.items(), key=lambda x: x[1], reverse=True)
    print("\n词性频率统计：")
    for pos, freq in sorted_pos_freq:
        print(f"{pos}: {freq}")

# 6. 生成词性词云
def generate_pos_wordcloud(pos_words, font_path):
    for pos, words in pos_words.items():
        if pos in ['n', 'v']:  # 只为名词和动词生成词云
            pos_type = "名词" if pos == 'n' else "动词"
            wc = wordcloud.WordCloud(font_path=font_path, background_color="white")
            wc.generate_from_frequencies(words)
            output_path = f"{pos}_wordcloud.png"
            wc.to_file(output_path)
            print(f"{pos_type}词云已生成：{output_path}")

# 7. Bigram分析，bigram（二元语法）是指文本中连续的两个词语组合，最终得到(n - 1)个二元组
# Counter()是collections模块提供的字典子类，用于高效统计可哈希对象的出现次数
def calculate_bigram_frequency(lines, stopwords):
    bigram_freq = Counter() # Counter对象，用于统计bigram的频率，格式为{bigram: 频率}
    for line in lines:
        words = [_ for _ in jieba.lcut(line) if _ not in stopwords] # 分词并去除停用词
        # 过滤掉包含特殊字符或单个字母的词
        words = [w for w in words if len(w) > 1 and w.isalnum()]
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        bigram_freq.update(bigrams)
    return bigram_freq

# 7. 打印top 10bigram
def plot_bigram(bigram_freq):
    print("\n频率最高的10个bigram：")
    for bigram, freq in bigram_freq.most_common(10):
        print(f"{bigram[0]}{bigram[1]}: {freq}")

    top_bigrams = bigram_freq.most_common(10)
    bigram_labels = [f"{b[0][0]}{b[0][1]}" for b in top_bigrams] # bigram标签，格式为"词1词2"
    bigram_values = [b[1] for b in top_bigrams]
    
    # 排除字体问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(bigram_labels))
    
    plt.barh(y_pos, bigram_values, align='center')
    plt.yticks(y_pos, bigram_labels)
    plt.xlabel('频率')
    plt.title('高频Bigram统计')
    
    plt.tight_layout()
    plt.savefig('bigram_freq.png')
    print("Bigram频率图已保存为：bigram_freq.png")

def main():

    font_path = "msyh.ttc"
    file_path = "week2.txt"
    stop_words_path = "stop_words.txt"
    # 1. 读取文本文件
    lines = read_file(file_path)
    
    # 4. 加载停用词
    stopwords = load_stopwords(stop_words_path)
    
    # 2. 3. 计算词频并打印top10
    word_freq = calculate_word_frequency(lines, stopwords)
    print_top_words(word_freq)
    
    # 5. 生成词云
    generate_wordcloud(word_freq, font_path, "wordcloud.png")
    
    # 6. 词性分析，画出n, v词云
    pos_freq, pos_words = analyze_pos(lines, stopwords)
    print_top_pos(pos_freq)
    generate_pos_wordcloud(pos_words, font_path)
    
    # 7. Bigram分析
    bigram_freq = calculate_bigram_frequency(lines, stopwords)
    plot_bigram(bigram_freq)

if __name__ == "__main__":
    main()

'''
利用python数据结构(list, set, Counter等)完成简单的词频分析任务。
具体地,基于提供的文本数据进行基本的词频统计。数据文件为week2.txt,一行为一个文本。
1. 读取文件,一行视为一个文档,打印出前10行观察这个文本是什么数据。
2. 使用jieba对所有文档(句子)进行分词,并统计词频。
3. 按词频进行排序,并输出词频最高的前10个词。
4. 引入停用词表(如"的"等语义不重要的词,可以自己定义,或者通过网络查找常用的停用词词表)进行停用词过滤,重新观察词频排序的结果。
5. 用wordcloud对高频词(部分低频词可以删除)进行可视化(词云)。
6. 对词性进行分析,观察不同词性的出现频率,并对特定词性的词进行可视化(词云)。
7. 如果用tuple来表示bigram,请统计所有的bigram的频率,并通过可视化观察高频的bigram。
8. 可否利用词频来进行特征词的筛选？如果有了特征词,怎么通过其来对文本进行向量表示？如果有了向量表示,可否计算不同句子之间的距离(相似性)？
'''
