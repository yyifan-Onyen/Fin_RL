import yfinance as yf
import pandas as pd

# 50只股票列表
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

print(f"📊 验证 {len(tickers)} 只股票数据...")

# 测试数据下载
start, end = '2023-01-01', '2024-12-31'
try:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, group_by='column')
    print(f"✅ 数据下载成功!")
    print(f"📈 数据形状: {raw.shape}")
    
    # 检查数据完整性
    close_data = raw.xs('Close', axis=1, level=0)
    print(f"📊 收盘价数据形状: {close_data.shape}")
    
    # 检查缺失数据
    missing_data = close_data.isnull().sum()
    problematic_stocks = missing_data[missing_data > 100]  # 缺失超过100天的股票
    
    if len(problematic_stocks) > 0:
        print(f"⚠️ 数据不完整的股票:")
        for stock, missing_days in problematic_stocks.items():
            print(f"  {stock}: 缺失 {missing_days} 天")
    else:
        print(f"✅ 所有股票数据完整!")
        
    # 计算买入持有收益率
    first_prices = close_data.iloc[0]
    last_prices = close_data.iloc[-1]
    returns = (last_prices / first_prices - 1) * 100
    
    print(f"\n📈 买入持有收益率统计:")
    print(f"平均收益率: {returns.mean():.2f}%")
    print(f"中位数收益率: {returns.median():.2f}%")
    print(f"最高收益率: {returns.max():.2f}% ({returns.idxmax()})")
    print(f"最低收益率: {returns.min():.2f}% ({returns.idxmin()})")
    
    # 等权重组合收益率
    equal_weight_return = returns.mean()
    print(f"🎯 等权重组合收益率: {equal_weight_return:.2f}%")
    
except Exception as e:
    print(f"❌ 数据下载失败: {e}") 