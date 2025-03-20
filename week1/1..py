import numpy as np
import matplotlib.pyplot as plt

def gaussian_distribution(x, mu, sigma2):
    """计算高斯分布概率密度"""
    sigma = np.sqrt(sigma2)
    return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-(x - mu)**2/(2*sigma2))

# 可调参数
mu = 0      # 均值
sigma2 = 1  # 方差

# 生成数据点
x = np.linspace(mu - 3*np.sqrt(sigma2), mu + 3*np.sqrt(sigma2), 200)
y = gaussian_distribution(x, mu, sigma2)

# 绘制曲线
plt.figure(figsize=(8,5))
plt.plot(x, y, 'b-', linewidth=2)
plt.title(f'Gaussian Distribution (μ={mu}, σ²={sigma2})')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
