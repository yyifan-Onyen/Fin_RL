# Fin_RL - 周逸凡 陈俊昊 第四组 强化学习 多臂老虎机强化学习投资组合策略

## 当前performance
### 多臂老虎机策略回测结果 (2023-2024)

🎯 **UCB (Upper Confidence Bound)**:
- 初始资金: $100,000.00
- 最终资金: $176,911.45
- 总收益: $76,911.45
- 收益率: 76.91%

🎯 **Epsilon-Greedy**:
- 初始资金: $100,000.00
- 最终资金: $176,939.46
- 总收益: $76,939.46
- 收益率: 76.94%

🎯 **Thompson Sampling**:
- 初始资金: $100,000.00
- 最终资金: $181,772.68
- 总收益: $81,772.68
- 收益率: 81.77%

**最佳策略**: Thompson Sampling 表现最优，收益率达到 81.77%


## 📋 项目概述

本项目包含传统技术分析策略和基于多臂老虎机的强化学习投资组合策略，用于股票交易回测。

## 📁 文件结构

```
Fin_RL/
├── portfolio_macd.py           # 原始MACD技术指标策略 赵huaqin的原始测试代码
├── portfolio_mab.py            # 多臂老虎机强化学习策略
├── test_mab_strategy.py        # 策略测试脚本
├── utils/                      # 工具包
│   ├── __init__.py
│   └── multi_armed_bandit_strategy.py  # 多臂老虎机策略实现
├── StockFormer/                # StockFormer深度学习框架
└── readme.md                   # 项目说明
```

## 策略实现

### 1. 多臂老虎机策略类型
- **UCB (Upper Confidence Bound)**: 平衡探索与利用
- **Epsilon-Greedy**: 简单的ε-贪婪策略  
- **Thompson Sampling**: 基于贝叶斯的概率采样

### 2. 主要特性
- 自适应投资组合权重分配
- 基于收益率的奖励机制
- 实时学习和策略更新
- 多策略性能比较

## 运行指令

### 1. 测试策略实现
```bash
python test_mab_strategy.py
```

### 2. 运行多臂老虎机回测
```bash
python portfolio_mab.py
```

### 3. 运行传统MACD策略
```bash
python portfolio_macd.py
```

## 输出结果

运行后会生成：
- `mab_strategy_comparison.png` - 策略性能对比图表
- `mab_strategy_results.csv` - 详细策略结果数据
- `portfolio_macd.png` - MACD策略图表

## 策略比较

| 策略类型 | 特点 | 适用场景 |
|---------|------|---------|
| MACD | 传统技术分析，基于移动平均 | 趋势跟踪 |
| UCB | 置信区间优化，平衡探索利用 | 不确定性较高的市场 |
| Epsilon-Greedy | 简单有效，固定探索率 | 稳定市场环境 |
| Thompson Sampling | 概率采样，贝叶斯更新 | 复杂多变的市场 |

## 重要信息

- **回测框架**: Backtrader
- **数据源**: Yahoo Finance (yfinance)
- **强化学习**: 自实现多臂老虎机算法
- **可视化**: Matplotlib
- **数据处理**: Pandas, NumPy

## TODO

- [x] 实现多臂老虎机策略
- [x] 创建策略比较框架
- [x] 添加可视化功能
- [ ] 集成StockFormer深度学习模型
- [ ] 添加更多技术指标
- [ ] 实现风险管理模块