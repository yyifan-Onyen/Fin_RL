"""
多臂老虎机强化学习策略模块
包含多种算法：Epsilon-Greedy, UCB, Thompson Sampling
用于股票组合交易决策
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats

class MultiArmedBanditStrategy(ABC):
    """多臂老虎机策略基类"""
    
    def __init__(self, n_arms: int, arm_names: List[str]):
        self.n_arms = n_arms
        self.arm_names = arm_names
        self.counts = np.zeros(n_arms)  # 每个臂被选择的次数
        self.values = np.zeros(n_arms)  # 每个臂的平均收益
        self.total_reward = 0
        self.total_counts = 0
        self.history = []  # 历史记录
        
    @abstractmethod
    def select_arm(self) -> int:
        """选择一个臂"""
        pass
    
    def update(self, chosen_arm: int, reward: float):
        """更新选择的臂的统计信息"""
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        self.total_reward += reward
        
        # 更新平均收益
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward
        
        # 记录历史
        self.history.append({
            'step': self.total_counts,
            'arm': chosen_arm,
            'arm_name': self.arm_names[chosen_arm],
            'reward': reward,
            'cumulative_reward': self.total_reward
        })
    
    def get_best_arm(self) -> int:
        """返回当前最佳臂"""
        return np.argmax(self.values)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / self.total_counts if self.total_counts > 0 else 0,
            'arm_values': dict(zip(self.arm_names, self.values)),
            'arm_counts': dict(zip(self.arm_names, self.counts)),
            'best_arm': self.arm_names[self.get_best_arm()]
        }


class EpsilonGreedyStrategy(MultiArmedBanditStrategy):
    """Epsilon-Greedy策略"""
    
    def __init__(self, n_arms: int, arm_names: List[str], epsilon: float = 0.1):
        super().__init__(n_arms, arm_names)
        self.epsilon = epsilon
    
    def select_arm(self) -> int:
        if np.random.random() > self.epsilon:
            # 利用：选择当前最佳臂
            return np.argmax(self.values)
        else:
            # 探索：随机选择
            return np.random.randint(self.n_arms)


class UCBStrategy(MultiArmedBanditStrategy):
    """Upper Confidence Bound策略"""
    
    def __init__(self, n_arms: int, arm_names: List[str], c: float = 1.0):
        super().__init__(n_arms, arm_names)
        self.c = c  # 探索参数
    
    def select_arm(self) -> int:
        # 如果有臂还未被选择过，优先选择
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # 计算UCB值
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            confidence_interval = self.c * np.sqrt(
                np.log(self.total_counts) / self.counts[arm]
            )
            ucb_values[arm] = self.values[arm] + confidence_interval
        
        return np.argmax(ucb_values)


class ThompsonSamplingStrategy(MultiArmedBanditStrategy):
    """Thompson Sampling策略（Beta分布）"""
    
    def __init__(self, n_arms: int, arm_names: List[str], alpha: float = 1.0, beta: float = 1.0):
        super().__init__(n_arms, arm_names)
        self.alpha = np.ones(n_arms) * alpha  # 成功参数
        self.beta = np.ones(n_arms) * beta    # 失败参数
    
    def select_arm(self) -> int:
        # 从每个臂的Beta分布中采样
        samples = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            samples[arm] = np.random.beta(self.alpha[arm], self.beta[arm])
        
        return np.argmax(samples)
    
    def update(self, chosen_arm: int, reward: float):
        """更新Beta分布参数"""
        super().update(chosen_arm, reward)
        
        # 将收益转换为成功/失败
        # 正收益视为成功，负收益视为失败
        if reward > 0:
            self.alpha[chosen_arm] += 1
        else:
            self.beta[chosen_arm] += 1


class StockMABStrategy:
    """股票多臂老虎机策略包装器"""
    
    def __init__(self, tickers: List[str], strategy_type: str = 'ucb', **kwargs):
        self.tickers = tickers
        self.n_stocks = len(tickers)
        
        # 选择策略
        if strategy_type == 'epsilon_greedy':
            self.strategy = EpsilonGreedyStrategy(self.n_stocks, tickers, **kwargs)
        elif strategy_type == 'ucb':
            self.strategy = UCBStrategy(self.n_stocks, tickers, **kwargs)
        elif strategy_type == 'thompson':
            self.strategy = ThompsonSamplingStrategy(self.n_stocks, tickers, **kwargs)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        self.current_positions = {ticker: 0 for ticker in tickers}
        self.price_history = {ticker: [] for ticker in tickers}
        self.last_prices: Dict[str, float] = {ticker: 0.0 for ticker in tickers}
        
    def select_stock(self) -> str:
        """选择要交易的股票"""
        arm = self.strategy.select_arm()
        return self.tickers[arm]
    
    def update_prices(self, prices: Dict[str, float]):
        """更新股票价格"""
        for ticker, price in prices.items():
            self.price_history[ticker].append(price)
            self.last_prices[ticker] = price
    
    def calculate_reward(self, ticker: str) -> float:
        """计算股票的收益率作为奖励"""
        if len(self.price_history[ticker]) < 2:
            return 0.0
        
        # 计算收益率
        current_price = self.price_history[ticker][-1]
        previous_price = self.price_history[ticker][-2]
        return (current_price - previous_price) / previous_price
    
    def update_strategy(self, ticker: str):
        """更新策略"""
        reward = self.calculate_reward(ticker)
        arm = self.tickers.index(ticker)
        self.strategy.update(arm, reward)
    
    def get_trading_signal(self, ticker: str, threshold: float = 0.02) -> str:
        """获取交易信号"""
        reward = self.calculate_reward(ticker)
        
        if reward > threshold:
            return 'BUY'
        elif reward < -threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def get_portfolio_weights(self) -> Dict[str, float]:
        """基于策略值获取投资组合权重"""
        values = self.strategy.values.copy()
        
        # 在学习期前几轮，使用策略特定的初始权重分配
        if self.strategy.total_counts < 10:  # 学习期
            if isinstance(self.strategy, EpsilonGreedyStrategy):  # Epsilon-Greedy
                # Epsilon-Greedy: 稍微偏向之前选择过的臂
                if self.strategy.total_counts > 0:
                    values = self.strategy.counts / (self.strategy.total_counts + 1e-8)
                else:
                    values = np.array([0.4, 0.3, 0.2, 0.1])  # 不同的初始偏好
            elif isinstance(self.strategy, UCBStrategy):  # UCB
                # UCB: 平衡探索，稍微不同的初始分配
                values = np.array([0.35, 0.25, 0.25, 0.15])
            elif isinstance(self.strategy, ThompsonSamplingStrategy):  # Thompson Sampling
                # Thompson Sampling: 基于Beta分布采样
                values = np.array([
                    np.random.beta(self.strategy.alpha[i], self.strategy.beta[i]) 
                    for i in range(self.n_stocks)
                ])
            else:
                values = np.ones(self.n_stocks) / self.n_stocks
        else:
            # 学习期后，使用实际的值
            # 添加一个小的基准值避免完全为0
            values = values + 0.001
            
            # 对于不同策略，使用不同的权重计算方法
            if isinstance(self.strategy, EpsilonGreedyStrategy):  # Epsilon-Greedy
                # 使用softmax转换，temperature参数控制集中度
                temperature = 2.0
                exp_values = np.exp(values / temperature)
                values = exp_values / np.sum(exp_values)
            elif isinstance(self.strategy, UCBStrategy):  # UCB
                # UCB使用置信区间影响权重
                if self.strategy.total_counts > 0:
                    ucb_values = np.zeros(self.n_stocks)
                    for i in range(self.n_stocks):
                        if self.strategy.counts[i] > 0:
                            confidence = self.strategy.c * np.sqrt(
                                np.log(self.strategy.total_counts) / self.strategy.counts[i]
                            )
                            ucb_values[i] = values[i] + confidence
                        else:
                            ucb_values[i] = values[i] + 1.0  # 未探索的臂给予高权重
                    values = ucb_values
                # Softmax转换
                values = np.maximum(values, 0) + 0.001
                values = values / np.sum(values)
            elif isinstance(self.strategy, ThompsonSamplingStrategy):  # Thompson Sampling
                # Thompson Sampling使用当前的采样值
                sampled_values = np.array([
                    np.random.beta(self.strategy.alpha[i], self.strategy.beta[i]) 
                    for i in range(self.n_stocks)
                ])
                values = sampled_values
        
        # 确保权重为正且和为1
        values = np.maximum(values, 0.001)  # 最小权重0.1%
        weights = values / np.sum(values)
        
        return dict(zip(self.tickers, weights))
    
    def get_statistics(self) -> Dict:
        """获取策略统计信息"""
        stats = self.strategy.get_statistics()
        stats['portfolio_weights'] = self.get_portfolio_weights()
        return stats
    
    def plot_performance(self, save_path: Optional[str] = None):
        """绘制性能图表"""
        if not self.strategy.history:
            print("No history to plot")
            return
        
        df = pd.DataFrame(self.strategy.history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 累积收益
        axes[0, 0].plot(df['step'], df['cumulative_reward'])
        axes[0, 0].set_title('Cumulative Reward')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Cumulative Reward')
        
        # 臂选择频率
        arm_counts = [self.strategy.counts[i] for i in range(self.n_stocks)]
        axes[0, 1].bar(self.tickers, arm_counts)
        axes[0, 1].set_title('Arm Selection Frequency')
        axes[0, 1].set_xlabel('Stocks')
        axes[0, 1].set_ylabel('Count')
        
        # 平均收益
        axes[1, 0].bar(self.tickers, self.strategy.values)
        axes[1, 0].set_title('Average Reward per Arm')
        axes[1, 0].set_xlabel('Stocks')
        axes[1, 0].set_ylabel('Average Reward')
        
        # 投资组合权重
        weights = self.get_portfolio_weights()
        axes[1, 1].pie(list(weights.values()), labels=list(weights.keys()), autopct='%1.1f%%')
        axes[1, 1].set_title('Portfolio Weights')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


# 使用示例和测试函数
def test_mab_strategy():
    """测试多臂老虎机策略"""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    # 测试不同策略
    strategies = {
        'Epsilon-Greedy': StockMABStrategy(tickers, 'epsilon_greedy', epsilon=0.1),
        'UCB': StockMABStrategy(tickers, 'ucb', c=1.0),
        'Thompson Sampling': StockMABStrategy(tickers, 'thompson')
    }
    
    # 模拟数据
    np.random.seed(42)
    n_steps = 100
    
    for name, strategy in strategies.items():
        print(f"\n=== Testing {name} ===")
        
        for step in range(n_steps):
            # 模拟股票价格变动
            prices = {}
            for ticker in tickers:
                # 模拟价格（随机游走 + 趋势）
                base_price = 100
                trend = {'AAPL': 0.001, 'MSFT': 0.0005, 'GOOGL': 0.0008, 'AMZN': 0.0003}
                noise = np.random.normal(0, 0.02)
                prices[ticker] = base_price * (1 + trend[ticker] + noise)
            
            strategy.update_prices(prices)
            
            # 选择股票并更新策略
            selected_stock = strategy.select_stock()
            strategy.update_strategy(selected_stock)
        
        # 输出结果
        stats = strategy.get_statistics()
        print(f"Total Reward: {stats['total_reward']:.4f}")
        print(f"Average Reward: {stats['average_reward']:.4f}")
        print(f"Best Stock: {stats['best_arm']}")
        print("Portfolio Weights:", {k: f"{v:.3f}" for k, v in stats['portfolio_weights'].items()})


if __name__ == "__main__":
    test_mab_strategy() 