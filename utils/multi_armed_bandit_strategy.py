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
        """计算股票的风险调整收益率作为奖励"""
        if len(self.price_history[ticker]) < 21:  # 需要至少21天数据
            return 0.0
        
        prices = np.array(self.price_history[ticker])
        
        # 1. 基础收益率
        current_price = prices[-1]
        previous_price = prices[-2]
        daily_return = (current_price - previous_price) / previous_price
        
        # 2. 短期动量 (5天)
        if len(prices) >= 5:
            momentum_5d = (prices[-1] - prices[-5]) / prices[-5]
        else:
            momentum_5d = 0.0
        
        # 3. 中期动量 (20天)
        if len(prices) >= 20:
            momentum_20d = (prices[-1] - prices[-20]) / prices[-20]
        else:
            momentum_20d = 0.0
        
        # 4. 波动率调整 (计算20天波动率)
        if len(prices) >= 20:
            returns = np.diff(prices[-20:]) / prices[-20:-1]
            volatility = np.std(returns)
            
            # 夏普比率式的风险调整
            if volatility > 0:
                risk_adjusted_return = daily_return / volatility
            else:
                risk_adjusted_return = daily_return
        else:
            risk_adjusted_return = daily_return
        
        # 5. 趋势强度 (使用移动平均)
        if len(prices) >= 10:
            ma_5 = np.mean(prices[-5:])
            ma_10 = np.mean(prices[-10:])
            trend_strength = (ma_5 - ma_10) / ma_10
        else:
            trend_strength = 0.0
        
        # 6. 相对强度 (与整体市场比较)
        # 计算该股票相对于所有股票的表现
        if len(self.price_history) > 1:
            all_returns = []
            for t in self.tickers:
                if len(self.price_history[t]) >= 2:
                    ret = (self.price_history[t][-1] - self.price_history[t][-2]) / self.price_history[t][-2]
                    all_returns.append(ret)
            
            if len(all_returns) > 0:
                market_return = np.mean(all_returns)
                relative_strength = daily_return - market_return
            else:
                relative_strength = 0.0
        else:
            relative_strength = 0.0
        
        # 7. 综合奖励计算 (加权组合)
        reward = (
            0.3 * risk_adjusted_return +      # 风险调整收益 (30%)
            0.2 * momentum_5d +               # 短期动量 (20%)
            0.1 * momentum_20d +              # 中期动量 (10%)
            0.2 * trend_strength +            # 趋势强度 (20%)
            0.2 * relative_strength           # 相对强度 (20%)
        )
        
        # 8. 奖励缩放和限制
        reward = np.tanh(reward * 10)  # 使用tanh限制在[-1, 1]
        
        return reward
    
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
        """基于策略值获取投资组合权重 - 改进版"""
        values = self.strategy.values.copy()
        
        # 延长学习期到50天，让策略有更多时间学习
        if self.strategy.total_counts < 50:  # 延长学习期
            if isinstance(self.strategy, EpsilonGreedyStrategy):  # Epsilon-Greedy
                # 使用更保守的初始分配，避免过度集中
                if self.strategy.total_counts > 0:
                    # 基于当前计数的softmax分配
                    counts = self.strategy.counts + 1  # 平滑处理
                    weights = counts / np.sum(counts)
                    # 加入温度参数降低集中度
                    temperature = 3.0
                    weights = np.power(weights, 1/temperature)
                    values = weights / np.sum(weights)
                else:
                    # 初始使用等权重分配
                    values = np.ones(self.n_stocks) / self.n_stocks
            elif isinstance(self.strategy, UCBStrategy):  # UCB
                # UCB策略在学习期使用更均匀的分配
                if self.strategy.total_counts > 0:
                    # 基于UCB值的软分配
                    ucb_values = np.zeros(self.n_stocks)
                    for i in range(self.n_stocks):
                        if self.strategy.counts[i] > 0:
                            confidence = self.strategy.c * np.sqrt(
                                np.log(self.strategy.total_counts) / self.strategy.counts[i]
                            )
                            ucb_values[i] = self.strategy.values[i] + confidence
                        else:
                            ucb_values[i] = 1.0  # 未探索的给予高权重
                    
                    # 使用softmax转换UCB值
                    exp_values = np.exp(ucb_values / 2.0)  # 温度参数2.0
                    values = exp_values / np.sum(exp_values)
                else:
                    values = np.ones(self.n_stocks) / self.n_stocks
            elif isinstance(self.strategy, ThompsonSamplingStrategy):  # Thompson Sampling
                # Thompson Sampling使用采样值
                sampled_values = np.array([
                    np.random.beta(self.strategy.alpha[i], self.strategy.beta[i]) 
                    for i in range(self.n_stocks)
                ])
                values = sampled_values / np.sum(sampled_values)
            else:
                values = np.ones(self.n_stocks) / self.n_stocks
        else:
            # 学习期后，使用更复杂的权重分配策略
            # 1. 基础值标准化
            values = values - np.min(values)  # 确保非负
            
            # 2. 不同策略的特定处理
            if isinstance(self.strategy, EpsilonGreedyStrategy):  # Epsilon-Greedy
                # 使用温度缩放的softmax
                temperature = 1.5
                exp_values = np.exp(values / temperature)
                values = exp_values / np.sum(exp_values)
            elif isinstance(self.strategy, UCBStrategy):  # UCB
                # UCB策略结合置信区间
                ucb_values = np.zeros(self.n_stocks)
                for i in range(self.n_stocks):
                    if self.strategy.counts[i] > 0:
                        confidence = self.strategy.c * np.sqrt(
                            np.log(self.strategy.total_counts) / self.strategy.counts[i]
                        )
                        ucb_values[i] = values[i] + confidence
                    else:
                        ucb_values[i] = values[i] + 0.5
                
                # 使用排名权重
                ranks = np.argsort(ucb_values)[::-1]  # 降序排列
                rank_weights = np.zeros(self.n_stocks)
                for i, rank in enumerate(ranks):
                    rank_weights[rank] = 1.0 / (i + 1)  # 排名权重
                
                values = rank_weights
            elif isinstance(self.strategy, ThompsonSamplingStrategy):  # Thompson Sampling
                # Thompson Sampling使用多次采样的平均
                samples = np.array([
                    [np.random.beta(self.strategy.alpha[i], self.strategy.beta[i]) 
                     for _ in range(10)]  # 采样10次取平均
                    for i in range(self.n_stocks)
                ])
                values = np.mean(samples, axis=1)
        
        # 3. 风险控制 - 限制单个股票的最大权重
        max_weight = 0.15  # 单个股票最大权重15%
        values = np.minimum(values, max_weight)
        
        # 4. 多样化约束 - 确保至少投资前80%的股票
        sorted_indices = np.argsort(values)[::-1]
        num_top_stocks = int(0.8 * self.n_stocks)
        min_weight = 0.005  # 最小权重0.5%
        
        for i in range(num_top_stocks):
            idx = sorted_indices[i]
            values[idx] = max(values[idx], min_weight)
        
        # 5. 最终标准化
        values = np.maximum(values, 0.001)  # 确保最小权重
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
    tickers = [
        # 困境股票 (6只) - 表现极差股票
        'PTON', 'PLUG', 'GOEV', 'BYND', 'RIVN', 'LCID',
        
        # 传统零售 (8只)
        'WMT', 'TGT', 'KSS', 'M', 'COST', 'HD', 'BBY', 'DG',
        
        # 能源股 (8只)  
        'XOM', 'CVX', 'BP', 'COP', 'SLB', 'HAL', 'OXY', 'DVN',
        
        # 传统银行 (6只)
        'BAC', 'WFC', 'C', 'JPM', 'USB', 'PNC',
        
        # 房地产 (6只)
        'SPG', 'EXR', 'PLD', 'AMT', 'O', 'VTR',
        
        # 传统制造/工业 (6只)
        'GE', 'F', 'GM', 'CAT', 'BA', 'MMM',
        
        # 传统媒体/电信 (5只)
        'DIS', 'CMCSA', 'VZ', 'T', 'PARA',
        
        # 传统航空 (5只)
        'DAL', 'AAL', 'UAL', 'LUV', 'ALK'
    ]
    
    print(f"📊 测试股票池: {len(tickers)} 只股票")
    
    # 测试不同策略
    strategies = {
        'Epsilon-Greedy': StockMABStrategy(tickers, 'epsilon_greedy', epsilon=0.1),
        'UCB': StockMABStrategy(tickers, 'ucb', c=1.0),
        'Thompson Sampling': StockMABStrategy(tickers, 'thompson')
    }
    
    # 模拟数据
    np.random.seed(42)
    n_steps = 200  # 增加步数以适应更多股票
    
    for name, strategy in strategies.items():
        print(f"\n=== Testing {name} ===")
        
        for step in range(n_steps):
            # 模拟股票价格变动
            prices = {}
            for i, ticker in enumerate(tickers):
                # 模拟价格（随机游走 + 不同的趋势）
                base_price = 100
                # 为不同类型股票设置不同的趋势
                trend_map = {
                    # 困境股票 - 负增长
                    'PTON': -0.0005, 'PLUG': -0.0008, 'GOEV': -0.0015,
                    'BYND': -0.0012, 'RIVN': -0.0010, 'LCID': -0.0008,
                    
                    # 传统零售 - 较低增长  
                    'WMT': 0.0002, 'TGT': 0.0001, 'KSS': -0.0001, 'M': -0.0003,
                    'COST': 0.0003, 'HD': 0.0002, 'BBY': 0.0000, 'DG': 0.0001,
                    
                    # 能源股 - 波动较大
                    'XOM': 0.0003, 'CVX': 0.0002, 'BP': 0.0001, 'COP': 0.0002,
                    'SLB': 0.0001, 'HAL': 0.0000, 'OXY': 0.0002, 'DVN': 0.0001,
                    
                    # 传统银行 - 中等增长
                    'BAC': 0.0002, 'WFC': 0.0001, 'C': 0.0001, 'JPM': 0.0003,
                    'USB': 0.0002, 'PNC': 0.0002,
                    
                    # 房地产 - 稳定增长
                    'SPG': 0.0001, 'EXR': 0.0004, 'PLD': 0.0005, 'AMT': 0.0004,
                    'O': 0.0003, 'VTR': 0.0002,
                    
                    # 传统制造 - 周期性
                    'GE': 0.0001, 'F': 0.0000, 'GM': 0.0001, 'CAT': 0.0002,
                    'BA': 0.0001, 'MMM': 0.0001,
                    
                    # 传统媒体/电信 - 下降趋势
                    'DIS': 0.0000, 'CMCSA': -0.0001, 'VZ': -0.0001, 'T': -0.0002,
                    'PARA': -0.0003,
                    
                    # 传统航空 - 波动较大
                    'DAL': 0.0001, 'AAL': -0.0001, 'UAL': 0.0000, 'LUV': 0.0001,
                    'ALK': 0.0000
                }
                trend = trend_map.get(ticker, 0.0000)
                noise = np.random.normal(0, 0.025)  # 增加波动性
                prices[ticker] = base_price * (1 + trend + noise)
            
            strategy.update_prices(prices)
            
            # 选择股票并更新策略
            selected_stock = strategy.select_stock()
            strategy.update_strategy(selected_stock)
        
        # 输出结果
        stats = strategy.get_statistics()
        print(f"Total Reward: {stats['total_reward']:.4f}")
        print(f"Average Reward: {stats['average_reward']:.4f}")
        print(f"Best Stock: {stats['best_arm']}")
        print("Portfolio Weights (top 10):")
        sorted_weights = sorted(stats['portfolio_weights'].items(), key=lambda x: x[1], reverse=True)
        for ticker, weight in sorted_weights[:10]:
            print(f"  {ticker}: {weight:.3f}")
        print("...")
        print(f"  Total stocks: {len(tickers)}")
        print(f"  Weight sum: {sum(stats['portfolio_weights'].values()):.3f}")


if __name__ == "__main__":
    test_mab_strategy() 