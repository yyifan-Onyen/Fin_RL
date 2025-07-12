"""
多臂老虎机强化学习策略模块
包含多种算法：Epsilon-Greedy, UCB, Thompson Sampling
用于股票组合交易决策
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
# import matplotlib.pyplot as plt
# from scipy import stats
import logging
import sys
import os

# 添加父目录到路径以导入alpaca模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

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
    
    def __init__(self, n_arms: int, arm_names: List[str], epsilon: float = 0.2):
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
    
    def __init__(self, n_arms: int, arm_names: List[str], c: float = 2.0):
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
    
    def __init__(self, n_arms: int, arm_names: List[str], alpha: float = 2.0, beta: float = 2.0):
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


class SmartMABTrader:
    """智能多臂老虎机交易员 - 核心功能类"""
    
    def __init__(self, alpaca_connection, strategy_type='ucb', max_stocks=50, **strategy_kwargs):
        """
        初始化智能MAB交易员
        
        Args:
            alpaca_connection: Alpaca API连接
            strategy_type: 策略类型 ('ucb', 'epsilon_greedy', 'thompson')
            max_stocks: 最大股票数量
            **strategy_kwargs: 策略参数
        """
        self.alpaca = alpaca_connection
        self.strategy_type = strategy_type
        self.max_stocks = max_stocks
        
        # 获取所有可交易股票
        logger.info("正在获取所有可交易股票...")
        all_stocks = self.alpaca.get_all_tradable_stocks()
        
        if not all_stocks:
            raise ValueError("无法获取可交易股票列表")
        
        # 智能选择股票池
        self.selected_stocks = self._select_stock_pool(all_stocks)
        logger.info(f"已选择 {len(self.selected_stocks)} 只股票用于MAB策略")
        
        # 初始化MAB策略
        self.mab_strategy = self._create_strategy(strategy_kwargs)
        
        # 价格历史和状态
        self.price_history = {ticker: [] for ticker in self.selected_stocks}
        self.last_prices = {ticker: 0.0 for ticker in self.selected_stocks}
        
        # 交易参数
        self.min_trade_amount = 100  # 最小交易金额
        self.learning_period = 50    # 学习期天数
        
        logger.info(f"SmartMABTrader初始化完成 - 策略: {strategy_type.upper()}")
    
    def _select_stock_pool(self, all_stocks: List[str]) -> List[str]:
        """
        智能选择股票池
        从所有可交易股票中选择最适合的股票
        """
        try:
            # 如果股票数量少于最大值，直接返回
            if len(all_stocks) <= self.max_stocks:
                return all_stocks
            
            # 选择策略：优先选择知名度高、流动性好的股票
            # 这里可以添加更复杂的筛选逻辑
            
            # 1. 优先选择大型股票（通常在列表前面）
            priority_stocks = []
            
            # 2. 添加知名科技股
            tech_stocks = [s for s in all_stocks if s in [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM'
            ]]
            priority_stocks.extend(tech_stocks)
            
            # 3. 添加知名金融股
            finance_stocks = [s for s in all_stocks if s in [
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF'
            ]]
            priority_stocks.extend(finance_stocks)
            
            # 4. 添加知名消费股
            consumer_stocks = [s for s in all_stocks if s in [
                'WMT', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'TGT', 'COST', 'LOW', 'PG'
            ]]
            priority_stocks.extend(consumer_stocks)
            
            # 5. 添加知名医疗股
            health_stocks = [s for s in all_stocks if s in [
                'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'DHR', 'BMY'
            ]]
            priority_stocks.extend(health_stocks)
            
            # 去重
            priority_stocks = list(set(priority_stocks))
            
            # 如果优先股票不够，随机选择其他股票
            if len(priority_stocks) < self.max_stocks:
                remaining_stocks = [s for s in all_stocks if s not in priority_stocks]
                np.random.shuffle(remaining_stocks)
                additional_needed = self.max_stocks - len(priority_stocks)
                priority_stocks.extend(remaining_stocks[:additional_needed])
            
            # 限制数量
            selected = priority_stocks[:self.max_stocks]
            
            logger.info(f"从 {len(all_stocks)} 只股票中选择了 {len(selected)} 只")
            return selected
            
        except Exception as e:
            logger.error(f"选择股票池失败: {e}")
            # 失败时返回前N只股票
            return all_stocks[:self.max_stocks]
    
    def _create_strategy(self, strategy_kwargs):
        """创建MAB策略"""
        n_stocks = len(self.selected_stocks)
        
        if self.strategy_type == 'epsilon_greedy':
            return EpsilonGreedyStrategy(n_stocks, self.selected_stocks, **strategy_kwargs)
        elif self.strategy_type == 'ucb':
            return UCBStrategy(n_stocks, self.selected_stocks, **strategy_kwargs)
        elif self.strategy_type == 'thompson':
            return ThompsonSamplingStrategy(n_stocks, self.selected_stocks, **strategy_kwargs)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
    
    def update_market_data(self):
        """更新市场数据"""
        try:
            # 获取当前价格
            current_prices = self.alpaca.get_current_prices(self.selected_stocks)
            
            if not current_prices:
                logger.warning("无法获取价格数据")
                return False
            
            # 更新价格历史
            for ticker, price in current_prices.items():
                self.price_history[ticker].append(price)
                self.last_prices[ticker] = price
            
            # 限制历史数据长度（保留最近100天）
            for ticker in self.selected_stocks:
                if len(self.price_history[ticker]) > 100:
                    self.price_history[ticker] = self.price_history[ticker][-100:]
            
            logger.info(f"更新了 {len(current_prices)} 只股票的价格数据")
            return True
            
        except Exception as e:
            logger.error(f"更新市场数据失败: {e}")
            return False
    
    def calculate_reward(self, ticker: str) -> float:
        """计算股票的多因子奖励"""
        if len(self.price_history[ticker]) < 21:
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
        
        # 4. 波动率调整
        if len(prices) >= 20:
            returns = np.diff(prices[-20:]) / prices[-20:-1]
            volatility = np.std(returns)
            
            if volatility > 0:
                risk_adjusted_return = daily_return / volatility
            else:
                risk_adjusted_return = daily_return
        else:
            risk_adjusted_return = daily_return
        
        # 5. 趋势强度
        if len(prices) >= 10:
            ma_5 = np.mean(prices[-5:])
            ma_10 = np.mean(prices[-10:])
            trend_strength = (ma_5 - ma_10) / ma_10
        else:
            trend_strength = 0.0
        
        # 6. 相对强度
        if len(self.price_history) > 1:
            all_returns = []
            for t in self.selected_stocks:
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
        
        # 7. 综合奖励计算
        reward = (
            0.3 * risk_adjusted_return +
            0.2 * momentum_5d +
            0.1 * momentum_20d +
            0.2 * trend_strength +
            0.2 * relative_strength
        )
        
        # 8. 奖励缩放
        reward = np.tanh(reward * 10)
        
        return reward
    
    def update_strategy(self):
        """更新MAB策略"""
        try:
            # 为每只股票计算奖励并更新策略
            for ticker in self.selected_stocks:
                if len(self.price_history[ticker]) >= 2:
                    reward = self.calculate_reward(ticker)
                    arm = self.selected_stocks.index(ticker)
                    self.mab_strategy.update(arm, reward)
            
            logger.info("MAB策略更新完成")
            
        except Exception as e:
            logger.error(f"更新策略失败: {e}")
    
    def get_trading_decisions(self) -> Dict:
        """获取交易决策"""
        try:
            # 获取账户信息
            account_info = self.alpaca.get_account_info()
            if not account_info:
                logger.error("无法获取账户信息")
                return {}
            
            total_equity = account_info['equity']
            
            # 获取投资组合权重
            portfolio_weights = self.get_portfolio_weights()
            
            # 获取当前持仓
            current_positions = self.alpaca.get_positions()
            position_dict = {}
            if not current_positions.empty:
                for _, pos in current_positions.iterrows():
                    position_dict[pos['symbol']] = {
                        'qty': pos['qty'],
                        'market_value': pos['market_value']
                    }
            
            # 计算交易决策
            trading_decisions = {
                'buy_orders': [],
                'sell_orders': [],
                'total_buy_value': 0,
                'total_sell_value': 0
            }
            
            for ticker, target_weight in portfolio_weights.items():
                if ticker not in self.last_prices or self.last_prices[ticker] <= 0:
                    continue
                
                target_value = total_equity * target_weight
                current_value = position_dict.get(ticker, {}).get('market_value', 0)
                current_price = self.last_prices[ticker]
                
                value_diff = target_value - current_value
                
                if abs(value_diff) > self.min_trade_amount:
                    if value_diff > 0:
                        # 买入
                        qty = int(value_diff / current_price)
                        if qty > 0:
                            trading_decisions['buy_orders'].append({
                                'symbol': ticker,
                                'qty': qty,
                                'price': current_price,
                                'value': qty * current_price
                            })
                            trading_decisions['total_buy_value'] += qty * current_price
                    else:
                        # 卖出
                        current_qty = position_dict.get(ticker, {}).get('qty', 0)
                        qty = min(int(abs(value_diff) / current_price), current_qty)
                        if qty > 0:
                            trading_decisions['sell_orders'].append({
                                'symbol': ticker,
                                'qty': qty,
                                'price': current_price,
                                'value': qty * current_price
                            })
                            trading_decisions['total_sell_value'] += qty * current_price
            
            return trading_decisions
            
        except Exception as e:
            logger.error(f"获取交易决策失败: {e}")
            return {}
    
    def get_portfolio_weights(self) -> Dict[str, float]:
        """获取投资组合权重"""
        values = self.mab_strategy.values.copy()
        
        # 学习期处理
        if self.mab_strategy.total_counts < self.learning_period:
            if isinstance(self.mab_strategy, EpsilonGreedyStrategy):
                if self.mab_strategy.total_counts > 0:
                    counts = self.mab_strategy.counts + 1
                    weights = counts / np.sum(counts)
                    temperature = 3.0
                    weights = np.power(weights, 1/temperature)
                    values = weights / np.sum(weights)
                else:
                    values = np.ones(len(self.selected_stocks)) / len(self.selected_stocks)
            elif isinstance(self.mab_strategy, UCBStrategy):
                if self.mab_strategy.total_counts > 0:
                    ucb_values = np.zeros(len(self.selected_stocks))
                    for i in range(len(self.selected_stocks)):
                        if self.mab_strategy.counts[i] > 0:
                            confidence = self.mab_strategy.c * np.sqrt(
                                np.log(self.mab_strategy.total_counts) / self.mab_strategy.counts[i]
                            )
                            ucb_values[i] = self.mab_strategy.values[i] + confidence
                        else:
                            ucb_values[i] = 1.0
                    
                    exp_values = np.exp(ucb_values / 2.0)
                    values = exp_values / np.sum(exp_values)
                else:
                    values = np.ones(len(self.selected_stocks)) / len(self.selected_stocks)
            elif isinstance(self.mab_strategy, ThompsonSamplingStrategy):
                sampled_values = np.array([
                    np.random.beta(self.mab_strategy.alpha[i], self.mab_strategy.beta[i]) 
                    for i in range(len(self.selected_stocks))
                ])
                values = sampled_values / np.sum(sampled_values)
            else:
                values = np.ones(len(self.selected_stocks)) / len(self.selected_stocks)
        else:
            # 学习期后的复杂权重分配
            values = values - np.min(values)
            
            if isinstance(self.mab_strategy, EpsilonGreedyStrategy):
                temperature = 1.5
                exp_values = np.exp(values / temperature)
                values = exp_values / np.sum(exp_values)
            elif isinstance(self.mab_strategy, UCBStrategy):
                ucb_values = np.zeros(len(self.selected_stocks))
                for i in range(len(self.selected_stocks)):
                    if self.mab_strategy.counts[i] > 0:
                        confidence = self.mab_strategy.c * np.sqrt(
                            np.log(self.mab_strategy.total_counts) / self.mab_strategy.counts[i]
                        )
                        ucb_values[i] = values[i] + confidence
                    else:
                        ucb_values[i] = values[i] + 0.5
                
                ranks = np.argsort(ucb_values)[::-1]
                rank_weights = np.zeros(len(self.selected_stocks))
                for i, rank in enumerate(ranks):
                    rank_weights[rank] = 1.0 / (i + 1)
                
                values = rank_weights
            elif isinstance(self.mab_strategy, ThompsonSamplingStrategy):
                samples = np.array([
                    [np.random.beta(self.mab_strategy.alpha[i], self.mab_strategy.beta[i]) 
                     for _ in range(10)]
                    for i in range(len(self.selected_stocks))
                ])
                values = np.mean(samples, axis=1)
        
        # 风险控制
        max_weight = 0.15
        values = np.minimum(values, max_weight)
        
        # 多样化约束
        sorted_indices = np.argsort(values)[::-1]
        num_top_stocks = int(0.8 * len(self.selected_stocks))
        min_weight = 0.005
        
        for i in range(num_top_stocks):
            idx = sorted_indices[i]
            values[idx] = max(values[idx], min_weight)
        
        # 最终标准化
        values = np.maximum(values, 0.001)
        weights = values / np.sum(values)
        
        return dict(zip(self.selected_stocks, weights))
    
    def execute_trading_plan(self, dry_run=True) -> Dict:
        """执行交易计划"""
        try:
            # 1. 更新市场数据
            if not self.update_market_data():
                logger.error("无法更新市场数据")
                return {}
            
            # 2. 更新策略
            self.update_strategy()
            
            # 3. 获取交易决策
            decisions = self.get_trading_decisions()
            
            if not decisions:
                logger.info("无交易决策")
                return {}
            
            # 4. 显示交易计划
            logger.info("=== 交易计划 ===")
            logger.info(f"买入订单: {len(decisions['buy_orders'])} 笔")
            logger.info(f"卖出订单: {len(decisions['sell_orders'])} 笔")
            logger.info(f"总买入价值: ${decisions['total_buy_value']:,.2f}")
            logger.info(f"总卖出价值: ${decisions['total_sell_value']:,.2f}")
            
            # 显示详细订单
            for order in decisions['buy_orders']:
                logger.info(f"买入: {order['symbol']} {order['qty']} 股 @ ${order['price']:.2f}")
            
            for order in decisions['sell_orders']:
                logger.info(f"卖出: {order['symbol']} {order['qty']} 股 @ ${order['price']:.2f}")
            
            # 5. 执行交易（如果不是演练模式）
            if not dry_run:
                executed_orders = []
                
                # 执行买入订单
                for order in decisions['buy_orders']:
                    result = self.alpaca.place_order(
                        symbol=order['symbol'],
                        qty=order['qty'],
                        side='buy'
                    )
                    if result:
                        executed_orders.append(result)
                
                # 执行卖出订单
                for order in decisions['sell_orders']:
                    result = self.alpaca.place_order(
                        symbol=order['symbol'],
                        qty=order['qty'],
                        side='sell'
                    )
                    if result:
                        executed_orders.append(result)
                
                logger.info(f"成功执行 {len(executed_orders)} 笔交易")
                decisions['executed_orders'] = executed_orders
            else:
                logger.info("演练模式 - 未实际执行交易")
            
            return decisions
            
        except Exception as e:
            logger.error(f"执行交易计划失败: {e}")
            return {}
    
    def get_strategy_status(self) -> Dict:
        """获取策略状态"""
        try:
            stats = self.mab_strategy.get_statistics()
            portfolio_weights = self.get_portfolio_weights()
            
            # 获取前10只股票
            top_stocks = sorted(portfolio_weights.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'strategy_type': self.strategy_type,
                'total_stocks': len(self.selected_stocks),
                'total_reward': stats['total_reward'],
                'average_reward': stats['average_reward'],
                'best_stock': stats['best_arm'],
                'learning_progress': min(self.mab_strategy.total_counts / self.learning_period, 1.0),
                'top_stocks': top_stocks,
                'portfolio_weights': portfolio_weights
            }
            
        except Exception as e:
            logger.error(f"获取策略状态失败: {e}")
            return {}


def create_smart_mab_trader(strategy_type='ucb', max_stocks=50, **strategy_kwargs):
    """
    创建智能MAB交易员
    
    Args:
        strategy_type: 策略类型 ('ucb', 'epsilon_greedy', 'thompson')
        max_stocks: 最大股票数量
        **strategy_kwargs: 策略参数
    """
    try:
        # 导入alpaca模块
        from alpaca import create_alpaca_connection
        
        # 创建Alpaca连接
        alpaca = create_alpaca_connection()
        if not alpaca:
            logger.error("无法创建Alpaca连接")
            return None
        
        # 创建智能MAB交易员
        trader = SmartMABTrader(
            alpaca_connection=alpaca,
            strategy_type=strategy_type,
            max_stocks=max_stocks,
            **strategy_kwargs
        )
        
        return trader
        
    except Exception as e:
        logger.error(f"创建智能MAB交易员失败: {e}")
        return None


# 测试函数
def test_smart_mab_trader():
    """测试智能MAB交易员"""
    print("🤖 智能MAB交易员测试")
    print("=" * 50)
    
    # 创建交易员
    trader = create_smart_mab_trader(
        strategy_type='ucb',
        max_stocks=30,
        c=2.0
    )
    
    if not trader:
        print("❌ 无法创建交易员")
        return
    
    print(f"✅ 交易员创建成功")
    print(f"📊 选择的股票: {trader.selected_stocks[:10]}...")
    
    # 获取策略状态
    status = trader.get_strategy_status()
    print(f"\n📈 策略状态:")
    print(f"   策略类型: {status.get('strategy_type', 'Unknown')}")
    print(f"   股票数量: {status.get('total_stocks', 0)}")
    print(f"   学习进度: {status.get('learning_progress', 0):.1%}")
    
    # 执行交易计划（演练模式）
    print(f"\n🎯 执行交易计划（演练模式）:")
    decisions = trader.execute_trading_plan(dry_run=True)
    
    if decisions:
        print(f"   买入订单: {len(decisions.get('buy_orders', []))}")
        print(f"   卖出订单: {len(decisions.get('sell_orders', []))}")
        print(f"   总买入价值: ${decisions.get('total_buy_value', 0):,.2f}")
        print(f"   总卖出价值: ${decisions.get('total_sell_value', 0):,.2f}")
    else:
        print("   无交易决策")


if __name__ == "__main__":
    test_smart_mab_trader() 