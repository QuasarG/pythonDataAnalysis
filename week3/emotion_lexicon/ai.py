import os
import re
import time
import requests
from pathlib import Path
from openai import OpenAI

class EmotionLexiconProcessor:
    def __init__(self, api_key, chunk_size=500, max_retries=3, retry_delay=2):
        self.chunk_size = chunk_size
        self.lexicon_dir = Path(__file__).parent
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def call_deepseek_api(self, prompt):
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat", # DeepSeek-V3
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"API调用失败: {str(e)}")
                time.sleep(self.retry_delay)

    def process_all_files(self):
        """处理所有情绪词典文件"""
        for file in self.lexicon_dir.glob('*.txt'):
            if file.name in ['anger.txt', 'disgust.txt', 'fear.txt', 'sadness.txt', 'joy.txt']:
                self.process_single_file(file)

    def process_single_file(self, file_path):
        """处理单个情绪文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        # 分块处理
        chunks = [lines[i:i+self.chunk_size] for i in range(0, len(lines), self.chunk_size)]
        
        output_dir = self.lexicon_dir / 'processed'
        output_dir.mkdir(exist_ok=True)
        
        # 处理每个分块并直接调用API
        for i, chunk in enumerate(chunks):
            prompt = (
                f"你是一个情感分析专家。请分析以下{file_path.stem}情绪类别的词语列表，并进行优化：\n"
                f"{','.join(chunk)}\n\n"
                "请按以下要求处理这些词语：\n"
                "1. 删除在网络评论中极少出现的书面语或文言词（如'怒发冲冠'）\n"
                "2. 为每个保留的词语提供1-3个在评论中常见的同义表达\n"
                "3. 添加在网络评论中经常出现、表达相同情绪的新词语\n\n"
                "请按此格式回复，每行一个词语：\n"
                "词语 同义词1 同义词2\n"
                "新词 同义词1 同义词2"
            )
            
            # 调用API获取结果
            result = self.call_deepseek_api(prompt)
            
            # 保存处理结果
            with open(output_dir / f"{file_path.stem}_processed.txt", 'a', encoding='utf-8') as f:
                f.write(result + '\n')

    @staticmethod
    def parse_results(result_text):
        """解析AI返回的结果"""
        results = []
        for line in result_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            words = line.split()
            if words:  # 只要有词就添加到结果中
                results.extend(words)
        
        return list(set(results))  # 去重

    def merge_results(self):
        """合并所有处理结果"""
        final_data = {}
        processed_dir = self.lexicon_dir / 'processed'
        
        # 合并所有_processed文件
        for result_file in processed_dir.glob('*_processed.txt'):
            emotion_type = result_file.stem.split('_')[0]
            with open(result_file, 'r', encoding='utf-8') as f:
                words = []
                for line in f:
                    words.extend(self.parse_results(line))
                for word in words:
                    final_data[word] = emotion_type

        # 保存最终结果
        with open(self.lexicon_dir / 'merged_lexicon.txt', 'w', encoding='utf-8') as f:
            for word, emotion in final_data.items():
                f.write(f"{word}||{emotion}\n")
        
        # 生成CSV表格
        import pandas as pd
        df = pd.DataFrame([
            {"keyword": word, "emotion": emotion}
            for word, emotion in final_data.items()
        ])
        df.to_csv(self.lexicon_dir / 'emotion_lexicon.csv', index=False)

if __name__ == '__main__':
    api_key = "xxxxxxxx"

    processor = EmotionLexiconProcessor(api_key)
    processor.process_all_files()
    processor.merge_results()
    print("处理完成！结果已保存到emotion_lexicon.csv和merged_lexicon.txt")