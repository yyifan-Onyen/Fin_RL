"""
portfolio_mab.py —— 基于多臂老虎机强化学习的投资组合策略
使用UCB、Epsilon-Greedy、Thompson Sampling等算法
"""

from datetime import datetime
import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('utils')
from multi_armed_bandit_strategy import StockMABStrategy


tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
start, end = '2023-01-01', '2024-12-31'
initial_cash = 100_000.0


print("📊 下载股票数据...")
raw = yf.download(
    tickers,
    start=start, end=end,
    auto_adjust=True,
    group_by='column'
)


lvl0 = list(raw.columns.get_level_values(0))
if set(tickers).issubset(lvl0):
    ticker_lvl, field_lvl = 0, 1   # ticker 在 Level-0
else:
    ticker_lvl, field_lvl = 1, 0   # ticker 在 Level-1


price_df = raw.xs('Close', axis=1, level=field_lvl).copy()
price_df.dropna(inplace=True)

print(f"📈 数据时间范围: {price_df.index[0]} 到 {price_df.index[-1]}")
print(f"📈 数据点数量: {len(price_df)}")

class MultiArmedBanditStrategy(bt.Strategy):
    
    params = dict(
        strategy_type='ucb',  # 默认策略类型
        rebalance_freq=5,     # 重新平衡频率（天）
        c=1.0,                # UCB参数
        epsilon=0.1,          # Epsilon-Greedy参数
        alpha=1.0,            # Thompson Sampling参数
        beta=1.0              # Thompson Sampling参数
    )

    def __init__(self):
        # 初始化多臂老虎机策略
        strategy_kwargs = {}
        if self.p.strategy_type == 'ucb':
            strategy_kwargs['c'] = self.p.c
        elif self.p.strategy_type == 'epsilon_greedy':
            strategy_kwargs['epsilon'] = self.p.epsilon
        elif self.p.strategy_type == 'thompson':
            strategy_kwargs['alpha'] = self.p.alpha
            strategy_kwargs['beta'] = self.p.beta
            
        self.mab_strategy = StockMABStrategy(
            tickers, 
            self.p.strategy_type,
            **strategy_kwargs
        )
        
        # 存储股票数据的引用
        self.stock_data = {}
        for i, d in enumerate(self.datas):
            ticker = tickers[i]
            self.stock_data[ticker] = d
            
        self.rebalance_days = []
        self.portfolio_values = []
        self.day_count = 0
        self.mab_decisions = []
        
        print(f"🤖 使用策略: {self.p.strategy_type.upper()}")

    def next(self):
        """每个交易日的策略逻辑"""
        self.day_count += 1
        
        # 需要足够的历史数据才开始交易
        if self.day_count < 20:  # 最少历史数据点
            return
        
        # 更新股票价格数据到MAB策略
        current_prices = {}
        for ticker in tickers:
            current_prices[ticker] = self.stock_data[ticker].close[0]
        
        self.mab_strategy.update_prices(current_prices)
        
        # 为每只股票更新策略（计算奖励）
        for ticker in tickers:
            self.mab_strategy.update_strategy(ticker)
        
        # 定期重新平衡
        if self.day_count % self.p.rebalance_freq == 0:
            self._rebalance_portfolio()

    def _rebalance_portfolio(self):
        """重新平衡投资组合"""
        current_value = self.broker.getvalue()
        current_cash = self.broker.getcash()
        
        # 获取MAB推荐的投资组合权重
        target_weights = self.mab_strategy.get_portfolio_weights()
        
        # 获取当前持仓
        current_positions = {}
        for ticker in tickers:
            pos = self.getposition(self.stock_data[ticker])
            current_positions[ticker] = pos.size
        
        print(f"\n📅 [{self.datetime.date(0)}] 重新平衡投资组合")
        print(f"💰 当前总价值: ${current_value:,.2f}")
        print(f"🎯 目标权重: {[f'{ticker}: {w:.3f}' for ticker, w in target_weights.items()]}")
        
        # 记录MAB决策
        stats = self.mab_strategy.get_statistics()
        self.mab_decisions.append({
            'date': self.datetime.date(0),
            'weights': target_weights.copy(),
            'best_stock': stats['best_arm'],
            'total_reward': stats['total_reward']
        })
        
        # 计算目标持仓
        for ticker, target_weight in target_weights.items():
            current_price = self.stock_data[ticker].close[0]
            target_value = current_value * target_weight
            target_shares = int(target_value / current_price)
            current_shares = current_positions[ticker]
            
            # 计算需要买入或卖出的股数
            diff_shares = target_shares - current_shares
            
            if abs(diff_shares) > 0:  # 只有在需要调整时才交易
                if diff_shares > 0:
                    # 买入
                    self.buy(data=self.stock_data[ticker], size=diff_shares)
                    print(f"  📈 买入 {ticker}: {diff_shares} 股 @ ${current_price:.2f}")
                else:
                    # 卖出
                    self.sell(data=self.stock_data[ticker], size=abs(diff_shares))
                    print(f"  📉 卖出 {ticker}: {abs(diff_shares)} 股 @ ${current_price:.2f}")
        
        self.rebalance_days.append(self.datetime.date(0))
        self.portfolio_values.append(current_value)

    def stop(self):
        """策略结束时的处理"""
        print(f"\n🎯 多臂老虎机策略统计:")
        stats = self.mab_strategy.get_statistics()
        
        print(f"📊 总奖励: {stats['total_reward']:.4f}")
        print(f"📊 平均奖励: {stats['average_reward']:.4f}")
        print(f"🏆 最佳股票: {stats['best_arm']}")
        print(f"�� 各股票平均奖励:")
        for ticker, value in stats['arm_values'].items():
            print(f"  {ticker}: {value:.4f}")
        
        print(f"📊 各股票选择次数:")
        for ticker, count in stats['arm_counts'].items():
            print(f"  {ticker}: {count}")


# ---------- 6. 策略比较类 ----------
class StrategyComparison:
    """比较不同多臂老虎机策略的性能"""
    
    def __init__(self, strategies=['ucb', 'epsilon_greedy', 'thompson']):
        self.strategies = strategies
        self.results = {}
    
    def run_strategy(self, strategy_type):
        """运行单个策略"""
        print(f"\n🚀 运行 {strategy_type.upper()} 策略...")
        
        # 为每个策略设置不同的随机种子
        strategy_seeds = {'ucb': 42, 'epsilon_greedy': 123, 'thompson': 456}
        np.random.seed(strategy_seeds.get(strategy_type, 42))
        
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)
        
        # 添加数据
        for i, ticker in enumerate(tickers):
            single = raw.xs(ticker, axis=1, level=ticker_lvl).dropna()
            single = single[['Open', 'High', 'Low', 'Close', 'Volume']]
            cerebro.adddata(bt.feeds.PandasData(dataname=single), name=ticker)
        
        # 添加策略并设置不同的参数
        strategy_params = {
            'ucb': {'c': 1.5},
            'epsilon_greedy': {'epsilon': 0.15}, 
            'thompson': {'alpha': 1.0, 'beta': 1.0}
        }
        
        cerebro.addstrategy(
            MultiArmedBanditStrategy, 
            strategy_type=strategy_type,
            **strategy_params.get(strategy_type, {})
        )
        
        # 运行
        result = cerebro.run()
        final_value = cerebro.broker.getvalue()
        
        print(f"✅ {strategy_type.upper()} 完成 - 最终价值: ${final_value:,.2f}")
        
        return {
            'strategy': strategy_type,
            'initial_value': initial_cash,
            'final_value': final_value,
            'total_return': final_value - initial_cash,
            'return_pct': (final_value - initial_cash) / initial_cash * 100,
            'cerebro': cerebro,
            'result': result[0]
        }
    
    def compare_strategies(self):
        """比较所有策略"""
        print("🔬 开始策略比较实验...")
        
        for strategy in self.strategies:
            self.results[strategy] = self.run_strategy(strategy)
        
        # 输出比较结果
        print(f"\n{'='*60}")
        print("📊 策略比较结果")
        print(f"{'='*60}")
        
        for strategy, result in self.results.items():
            print(f"\n🎯 {strategy.upper()}:")
            print(f"  初始资金: ${result['initial_value']:,.2f}")
            print(f"  最终资金: ${result['final_value']:,.2f}")
            print(f"  总收益: ${result['total_return']:,.2f}")
            print(f"  收益率: {result['return_pct']:.2f}%")
        
        # 找出最佳策略
        best_strategy = max(self.results.items(), key=lambda x: x[1]['final_value'])
        print(f"\n🏆 最佳策略: {best_strategy[0].upper()}")
        print(f"🏆 最佳收益率: {best_strategy[1]['return_pct']:.2f}%")
        
        return self.results


# ---------- 7. 运行主程序 ----------
def main():
    """主程序"""
    print("🤖 多臂老虎机投资组合策略回测")
    print(f"📅 回测期间: {start} 到 {end}")
    print(f"📈 股票池: {', '.join(tickers)}")
    print(f"💰 初始资金: ${initial_cash:,.2f}")
    
    # 运行策略比较
    comparison = StrategyComparison()
    results = comparison.compare_strategies()
    
    # 生成图表
    print(f"\n📊 生成性能图表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制收益率比较
    strategies = list(results.keys())
    returns = [results[s]['return_pct'] for s in strategies]
    
    axes[0, 0].bar(strategies, returns)
    axes[0, 0].set_title('不同策略收益率比较')
    axes[0, 0].set_ylabel('收益率 (%)')
    
    # 绘制最终价值比较
    final_values = [results[s]['final_value'] for s in strategies]
    axes[0, 1].bar(strategies, final_values)
    axes[0, 1].set_title('最终投资组合价值')
    axes[0, 1].set_ylabel('价值 ($)')
    
    # 绘制股票价格走势
    for ticker in tickers:
        axes[1, 0].plot(price_df.index, price_df[ticker], label=ticker)
    axes[1, 0].set_title('股票价格走势')
    axes[1, 0].legend()
    
    # 绘制基准比较（等权重买入持有）
    equal_weights = {ticker: 0.25 for ticker in tickers}
    benchmark_returns = []
    for date in price_df.index:
        daily_return = sum(
            equal_weights[ticker] * price_df.loc[date, ticker] / price_df.iloc[0][ticker]
            for ticker in tickers
        )
        benchmark_returns.append(daily_return * initial_cash)
    
    axes[1, 1].plot(price_df.index, benchmark_returns, label='等权重基准', linestyle='--')
    axes[1, 1].set_title('策略 vs 基准')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('mab_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 图表已保存为: mab_strategy_comparison.png")
    
    # 保存详细结果
    results_df = pd.DataFrame({
        strategy: {
            'Initial Value': result['initial_value'],
            'Final Value': result['final_value'],
            'Total Return': result['total_return'],
            'Return Percentage': result['return_pct']
        }
        for strategy, result in results.items()
    }).T
    
    results_df.to_csv('mab_strategy_results.csv')
    print("📊 详细结果已保存为: mab_strategy_results.csv")


if __name__ == "__main__":
    main() 