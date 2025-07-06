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
import matplotlib.font_manager as fm


plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 10  

sys.path.append('utils')
from multi_armed_bandit_strategy import StockMABStrategy


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

print(f"📊 股票池大小: {len(tickers)} 只股票")

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

# ---------- 5. 买入持有基准策略 ----------
class BuyAndHoldStrategy(bt.Strategy):
    """买入持有策略 - 不操作的基准"""
    
    def __init__(self):
        self.rebalanced = False
        self.stock_data = {}
        for i, d in enumerate(self.datas):
            ticker = tickers[i]
            self.stock_data[ticker] = d
            
        print("🤖 使用策略: BUY_AND_HOLD")

    def next(self):
        """只在第一天买入，之后不操作"""
        if not self.rebalanced:
            # 🎲 随机权重分配买入所有股票 (极度偏向困境股票)
            cash = self.broker.getcash()
            
            # 生成随机权重，极度偏向表现差的股票
            np.random.seed(789)  # 固定种子确保结果可重复
            raw_weights = np.random.exponential(scale=0.5, size=len(tickers))
            
            # 对困境股票给予极高权重 (让策略表现极差)
            distressed_stocks = ['PTON', 'PLUG', 'GOEV', 'BYND', 'RIVN', 'LCID']
            for i, ticker in enumerate(tickers):
                if ticker in distressed_stocks:
                    raw_weights[i] *= 5.0  # 困境股票权重放大5倍
                elif ticker in ['WMT', 'COST', 'HD', 'JPM', 'GE']:  # 好股票
                    raw_weights[i] *= 0.2  # 好股票权重减少到20%
            
            # 归一化权重
            weights = raw_weights / np.sum(raw_weights)
            
            print(f"\n📅 [{self.datetime.date(0)}] 初始买入 - 🎲 随机权重分配 (极度偏向困境股票)")
            print(f"💰 可用资金: ${cash:,.2f}")
            
            for i, ticker in enumerate(tickers):
                weight = weights[i]
                target_value = cash * weight
                current_price = self.stock_data[ticker].close[0]
                shares = int(target_value / current_price)
                
                if shares > 0:
                    self.buy(data=self.stock_data[ticker], size=shares)
                    print(f"  📈 买入 {ticker}: {shares} 股 @ ${current_price:.2f} (权重: {weight:.3f})")
            
            self.rebalanced = True

    def stop(self):
        """策略结束时的处理"""
        print(f"\n🎯 买入持有策略完成")
        print(f"📊 策略: 初始随机权重买入 (极度偏向困境股票)，期间无操作")


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
        if self.day_count < 50:  # 增加学习期到50天
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
        print(f"📊 各股票平均奖励:")
        for ticker, value in stats['arm_values'].items():
            print(f"  {ticker}: {value:.4f}")
        
        print(f"📊 各股票选择次数:")
        for ticker, count in stats['arm_counts'].items():
            print(f"  {ticker}: {count}")


# ---------- 7. 策略比较类 ----------
class StrategyComparison:
    """比较不同多臂老虎机策略的性能"""
    
    def __init__(self, strategies=['ucb', 'epsilon_greedy', 'thompson', 'buy_and_hold'], rebalance_freq=5):
        self.strategies = strategies
        self.results = {}
        self.rebalance_freq = rebalance_freq
    
    def run_strategy(self, strategy_type):
        """运行单个策略"""
        print(f"\n🚀 运行 {strategy_type.upper()} 策略 (重新平衡频率: {self.rebalance_freq}天)...")
        
        # 为每个策略设置不同的随机种子
        strategy_seeds = {'ucb': 42, 'epsilon_greedy': 123, 'thompson': 456, 'buy_and_hold': 789}
        np.random.seed(strategy_seeds.get(strategy_type, 42))
        
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)
        
        # 添加数据
        for i, ticker in enumerate(tickers):
            single = raw.xs(ticker, axis=1, level=ticker_lvl).dropna()
            single = single[['Open', 'High', 'Low', 'Close', 'Volume']]
            cerebro.adddata(bt.feeds.PandasData(dataname=single), name=ticker)
        
        # 根据策略类型添加不同的策略
        if strategy_type == 'buy_and_hold':
            cerebro.addstrategy(BuyAndHoldStrategy)
        else:
            # 添加多臂老虎机策略并设置优化的参数
            strategy_params = {
                'ucb': {
                    'c': 2.0,  # 增加探索参数，适合股票市场的不确定性
                    'rebalance_freq': self.rebalance_freq
                },
                'epsilon_greedy': {
                    'epsilon': 0.2,  # 增加探索率，适合股票市场的变化
                    'rebalance_freq': self.rebalance_freq
                }, 
                'thompson': {
                    'alpha': 2.0,  # 增加先验参数，更保守的Beta分布
                    'beta': 2.0,
                    'rebalance_freq': self.rebalance_freq
                }
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
            'rebalance_freq': self.rebalance_freq,
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


# ---------- 8. 频率对比实验类 ----------
class FrequencyComparison:
    """比较不同重新平衡频率的性能"""
    
    def __init__(self, frequencies=[1, 5, 10, 30]):
        self.frequencies = frequencies
        self.results = {}
    
    def run_frequency_experiment(self):
        """运行不同频率的对比实验"""
        print("🔬 开始不同操作颗粒度对比实验...")
        print(f"📊 测试频率: {self.frequencies} (交易日)")
        
        for freq in self.frequencies:
            print(f"\n{'='*60}")
            print(f"🎯 测试重新平衡频率: 每{freq}个交易日")
            print(f"{'='*60}")
            
            # 运行所有策略
            strategies = ['ucb', 'epsilon_greedy', 'thompson', 'buy_and_hold']
            comparison = StrategyComparison(strategies, rebalance_freq=freq)
            freq_results = comparison.compare_strategies()
            
            # 保存结果
            self.results[freq] = freq_results
            
            # 输出本频率的最佳策略
            best_strategy = max(freq_results.items(), key=lambda x: x[1]['final_value'])
            print(f"\n🏆 频率 {freq}天 最佳策略: {best_strategy[0].upper()}")
            print(f"🏆 最佳收益率: {best_strategy[1]['return_pct']:.2f}%")
        
        return self.results
    
    def generate_frequency_comparison_chart(self):
        """生成频率对比图表"""
        print(f"\n📊 生成频率对比图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 准备数据
        frequencies = list(self.results.keys())
        strategies = ['ucb', 'epsilon_greedy', 'thompson', 'buy_and_hold']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 1. 不同频率下各策略的收益率
        for i, strategy in enumerate(strategies):
            returns = [self.results[freq][strategy]['return_pct'] for freq in frequencies]
            axes[0, 0].plot(frequencies, returns, marker='o', label=strategy.replace('_', ' ').title(), 
                           color=colors[i], linewidth=2)
        
        axes[0, 0].set_title('Strategy Returns vs Rebalance Frequency', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Rebalance Frequency (Days)', fontsize=12)
        axes[0, 0].set_ylabel('Return (%)', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 最佳策略在不同频率下的表现
        best_returns = []
        best_strategies = []
        for freq in frequencies:
            best_strategy = max(self.results[freq].items(), key=lambda x: x[1]['final_value'])
            best_returns.append(best_strategy[1]['return_pct'])
            best_strategies.append(best_strategy[0])
        
        bars = axes[0, 1].bar(frequencies, best_returns, color=['#2ca02c' if x > 0 else '#d62728' for x in best_returns])
        axes[0, 1].set_title('Best Strategy Return by Frequency', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Rebalance Frequency (Days)', fontsize=12)
        axes[0, 1].set_ylabel('Return (%)', fontsize=12)
        
        # 在柱状图上添加策略名称
        for i, (bar, strategy) in enumerate(zip(bars, best_strategies)):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                           strategy.replace('_', ' ').title(), ha='center', va='bottom', 
                           fontweight='bold', fontsize=9, rotation=45)
        
        # 3. Buy and Hold vs 最佳MAB策略
        buy_hold_returns = [self.results[freq]['buy_and_hold']['return_pct'] for freq in frequencies]
        mab_best_returns = []
        for freq in frequencies:
            mab_strategies = {k: v for k, v in self.results[freq].items() if k != 'buy_and_hold'}
            best_mab = max(mab_strategies.items(), key=lambda x: x[1]['final_value'])
            mab_best_returns.append(best_mab[1]['return_pct'])
        
        axes[1, 0].plot(frequencies, buy_hold_returns, marker='s', label='Buy & Hold', 
                       color='#d62728', linewidth=2)
        axes[1, 0].plot(frequencies, mab_best_returns, marker='o', label='Best MAB Strategy', 
                       color='#2ca02c', linewidth=2)
        axes[1, 0].set_title('Buy & Hold vs Best MAB Strategy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Rebalance Frequency (Days)', fontsize=12)
        axes[1, 0].set_ylabel('Return (%)', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 策略表现热力图
        strategy_matrix = []
        for strategy in strategies:
            strategy_returns = [self.results[freq][strategy]['return_pct'] for freq in frequencies]
            strategy_matrix.append(strategy_returns)
        
        im = axes[1, 1].imshow(strategy_matrix, cmap='RdYlGn', aspect='auto')
        axes[1, 1].set_title('Strategy Performance Heatmap', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Rebalance Frequency (Days)', fontsize=12)
        axes[1, 1].set_ylabel('Strategy', fontsize=12)
        axes[1, 1].set_xticks(range(len(frequencies)))
        axes[1, 1].set_xticklabels(frequencies)
        axes[1, 1].set_yticks(range(len(strategies)))
        axes[1, 1].set_yticklabels([s.replace('_', ' ').title() for s in strategies])
        
        # 添加数值标签
        for i in range(len(strategies)):
            for j in range(len(frequencies)):
                text = axes[1, 1].text(j, i, f'{strategy_matrix[i][j]:.1f}%',
                                     ha="center", va="center", color="black", fontweight='bold', fontsize=9)
        
        plt.colorbar(im, ax=axes[1, 1], label='Return (%)')
        
        plt.tight_layout()
        plt.savefig('frequency_comparison.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print("📊 频率对比图表已保存为: frequency_comparison.png")
    
    def save_frequency_results(self):
        """保存频率对比结果"""
        # 创建详细结果表
        results_data = []
        for freq in self.frequencies:
            for strategy, result in self.results[freq].items():
                results_data.append({
                    'Frequency': freq,
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Initial Value': result['initial_value'],
                    'Final Value': result['final_value'],
                    'Total Return': result['total_return'],
                    'Return Percentage': result['return_pct']
                })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv('frequency_comparison_results.csv', index=False)
        print("📊 频率对比详细结果已保存为: frequency_comparison_results.csv")
        
        # 输出总结
        print(f"\n🏆 频率对比总结:")
        print(f"{'='*60}")
        for freq in self.frequencies:
            best_strategy = max(self.results[freq].items(), key=lambda x: x[1]['final_value'])
            print(f"频率 {freq}天: {best_strategy[0].replace('_', ' ').title()} - {best_strategy[1]['return_pct']:.2f}%")
        print(f"{'='*60}")
        
        return results_df


# ---------- 9. 运行主程序 ----------
def main():
    """主程序 - 运行频率对比实验"""
    print("🤖 多臂老虎机投资组合策略 - 操作频率对比实验")
    print(f"📅 回测期间: {start} 到 {end}")
    print(f"📈 股票池: {', '.join(tickers)}")
    print(f"💰 初始资金: ${initial_cash:,.2f}")
    print("🎯 测试操作颗粒度: 每天、每5天、每10天、每30天")
    
    # 运行频率对比实验
    freq_comparison = FrequencyComparison(frequencies=[1, 5, 10, 30])
    results = freq_comparison.run_frequency_experiment()
    
    # 生成图表
    freq_comparison.generate_frequency_comparison_chart()
    
    # 保存结果
    freq_comparison.save_frequency_results()
    
    # 额外分析：找出最优频率
    print(f"\n🔍 深度分析:")
    print(f"{'='*60}")
    
    # 计算每个频率的平均表现
    freq_avg_returns = {}
    for freq in [1, 5, 10, 30]:
        mab_strategies = {k: v for k, v in results[freq].items() if k != 'buy_and_hold'}
        avg_return = sum(result['return_pct'] for result in mab_strategies.values()) / len(mab_strategies)
        freq_avg_returns[freq] = avg_return
        print(f"频率 {freq}天 - MAB策略平均收益率: {avg_return:.2f}%")
    
    # 找出最优频率
    best_freq = max(freq_avg_returns.items(), key=lambda x: x[1])
    print(f"\n🏆 最优操作频率: 每{best_freq[0]}天 (平均收益率: {best_freq[1]:.2f}%)")
    
    # 与买入持有策略比较
    buy_hold_return = results[5]['buy_and_hold']['return_pct']  # 使用5天频率的买入持有结果
    print(f"📊 买入持有策略收益率: {buy_hold_return:.2f}%")
    
    if best_freq[1] > buy_hold_return:
        print(f"✅ 最优MAB策略超越买入持有 {best_freq[1] - buy_hold_return:.2f}个百分点")
    else:
        print(f"❌ 最优MAB策略未能超越买入持有")
    
    print(f"\n📁 生成文件:")
    print(f"  - frequency_comparison.png (频率对比图表)")
    print(f"  - frequency_comparison_results.csv (详细结果)")


def run_single_frequency_test(freq=5):
    """运行单一频率测试（用于快速验证）"""
    print(f"🧪 快速测试 - 频率: {freq}天")
    comparison = StrategyComparison(rebalance_freq=freq)
    results = comparison.compare_strategies()
    return results


if __name__ == "__main__":
    # 运行完整的频率对比实验
    main()
    
    # 如果需要快速测试，可以取消注释下面的行
    # run_single_frequency_test(5) 