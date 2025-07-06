"""
测试多臂老虎机策略的快速验证脚本
"""

import numpy as np
import sys
import os

# 添加utils路径
sys.path.append('utils')

def test_mab_basic():
    """基础功能测试"""
    print("🧪 开始基础功能测试...")
    
    try:
        from multi_armed_bandit_strategy import StockMABStrategy
        
        # 测试策略初始化
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        # 测试UCB策略
        strategy = StockMABStrategy(tickers, 'ucb', c=1.0)
        print("✅ UCB策略初始化成功")
        
        # 模拟一些价格数据
        np.random.seed(42)
        for step in range(10):
            # 模拟价格
            prices = {ticker: 100 + np.random.normal(0, 5) for ticker in tickers}
            strategy.update_prices(prices)
            
            if step > 0:  # 需要至少2个价格点才能计算收益
                # 选择股票
                selected = strategy.select_stock()
                print(f"第{step}步: 选择 {selected}")
                
                # 更新策略
                strategy.update_strategy(selected)
        
        # 获取统计信息
        stats = strategy.get_statistics()
        print(f"\n📊 最终统计:")
        print(f"  最佳股票: {stats['best_arm']}")
        print(f"  总奖励: {stats['total_reward']:.4f}")
        print(f"  平均奖励: {stats['average_reward']:.4f}")
        
        # 获取投资组合权重
        weights = strategy.get_portfolio_weights()
        print(f"  投资组合权重: {weights}")
        
        print("✅ 基础功能测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_all_strategies():
    """测试所有策略类型"""
    print("\n🧪 测试所有策略类型...")
    
    try:
        from multi_armed_bandit_strategy import StockMABStrategy
        
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        strategies = {
            'UCB': StockMABStrategy(tickers, 'ucb', c=1.0),
            'Epsilon-Greedy': StockMABStrategy(tickers, 'epsilon_greedy', epsilon=0.1),
            'Thompson Sampling': StockMABStrategy(tickers, 'thompson')
        }
        
        np.random.seed(42)
        
        for name, strategy in strategies.items():
            print(f"\n测试 {name} 策略:")
            
            # 模拟交易
            for step in range(20):
                prices = {ticker: 100 * (1 + 0.001 * step + np.random.normal(0, 0.02)) 
                         for ticker in tickers}
                strategy.update_prices(prices)
                
                if step > 0:
                    selected = strategy.select_stock()
                    strategy.update_strategy(selected)
            
            # 输出结果
            stats = strategy.get_statistics()
            print(f"  ✅ 最佳股票: {stats['best_arm']}")
            print(f"  ✅ 总奖励: {stats['total_reward']:.4f}")
        
        print("✅ 所有策略测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 策略测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 多臂老虎机策略测试")
    print("="*50)
    
    # 运行测试
    test1 = test_mab_basic()
    test2 = test_all_strategies()
    
    if test1 and test2:
        print("\n🎉 所有测试通过! 策略实现正确!")
        print("\n📋 接下来你可以:")
        print("1. 运行 python portfolio_mab.py 进行完整回测")
        print("2. 比较与原始MACD策略的性能")
        print("3. 调整策略参数进行优化")
    else:
        print("\n❌ 部分测试失败，请检查实现")

if __name__ == "__main__":
    main() 