import pandas as pd

# 创建一个包含日期字符串的DataFrame
data = {'comment_time': ['2023-01-01 12:00', '2023-02-15 18:30', '2023-03-20 09:45']}
df = pd.DataFrame(data)

# 使用pd.to_datetime将comment_time列转换为datetime类型
df['comment_time'] = pd.to_datetime(df['comment_time'])

# 输出处理后的DataFrame
print(df)