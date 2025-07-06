# portfolio_macd.py  —— robust & self-detecting version + P/L printout
from datetime import datetime
import backtrader as bt
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- 1. 组合配置 ----------
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
weights  = {t: 1/len(tickers) for t in tickers}
start, end = '2023-01-01', '2024-12-31'

# ---------- 2. 下载数据 ----------
raw = yf.download(
    tickers,
    start=start, end=end,
    auto_adjust=True,
    group_by='column'
)

# ---------- 3. 判断列层级顺序 ----------
lvl0 = list(raw.columns.get_level_values(0))
if set(tickers).issubset(lvl0):
    ticker_lvl, field_lvl = 0, 1   # ticker 在 Level-0
else:
    ticker_lvl, field_lvl = 1, 0   # ticker 在 Level-1

# ---------- 4. 取复权价格 (Close) ----------
price_df = raw.xs('Close', axis=1, level=field_lvl).copy()
price_df.dropna(inplace=True)

# ---------- 5. 组合净值曲线 ----------
normed    = price_df / price_df.iloc[0]
portfolio = (normed * pd.Series(weights)).sum(axis=1)

port_df = pd.DataFrame({
    'Open':  portfolio,
    'High':  portfolio,
    'Low':   portfolio,
    'Close': portfolio,
    'Volume': 0,
})
port_df.index.name = 'Date'

# ---------- 6. 策略 ----------
class WeightedBasketMACD(bt.Strategy):
    params = dict(weights=weights, max_leverage=0.99)

    def __init__(self):
        self.macd      = bt.ind.MACD(self.datas[0].close)
        self.crossup   = bt.ind.CrossUp(self.macd.macd, self.macd.signal)
        self.crossdown = bt.ind.CrossDown(self.macd.macd, self.macd.signal)
        self.stock_data = {d._name: d for d in self.datas[1:]}

    def _shares(self, price, cash):
        return int(cash / price)

    def next(self):
        cash = self.broker.getcash()

        if self.crossup[0]:                                    # 金叉——买
            invest_cash = cash * self.p.max_leverage
            print(f"[{self.datetime.date(0)}] BUY basket ${invest_cash:,.2f}")
            for t, w in self.p.weights.items():
                d, price = self.stock_data[t], self.stock_data[t].close[0]
                size = self._shares(price, invest_cash * w)
                if size > 0:
                    self.buy(data=d, size=size)
                    print(f"  -> buy {size:5d} {t} @ {price:.2f}")

        elif self.crossdown[0]:                                 # 死叉——卖
            print(f"[{self.datetime.date(0)}] SELL basket")
            for t, d in self.stock_data.items():
                pos = self.getposition(d)
                if pos.size:
                    self.close(data=d)
                    print(f"  -> sell {t} ({pos.size} shares)")

# ---------- 7. 构建 Cerebro ----------
initial_cash = 100_000.0
cerebro = bt.Cerebro()
cerebro.broker.setcash(initial_cash)
cerebro.broker.setcommission(commission=0.001)

# 7.1 组合净值 feed
cerebro.adddata(bt.feeds.PandasData(dataname=port_df), name='PORT')

# 7.2 每只股票 feed
for t in tickers:
    single = raw.xs(t, axis=1, level=ticker_lvl).dropna()
    single = single[['Open', 'High', 'Low', 'Close', 'Volume']]
    cerebro.adddata(bt.feeds.PandasData(dataname=single), name=t)

cerebro.addstrategy(WeightedBasketMACD)

# ---------- 8. 运行 ----------
cerebro.run()

# ---------- 9. 结果输出 ----------
final_value = cerebro.broker.getvalue()
profit      = final_value - initial_cash
return_pct  = profit / initial_cash * 100
print(f"\n===== 回测结束 =====")
print(f"初始资金: ${initial_cash:,.2f}")
print(f"最终资金: ${final_value:,.2f}")
print(f"净收益  : ${profit:,.2f}  ({return_pct:.2f}%)")

# ---------- 10. 绘图 ----------
fig = cerebro.plot(style='candlestick')[0][0]
fig.savefig('portfolio_macd.png')
print("已生成组合 MACD 图：portfolio_macd.png")
