#!/usr/bin/env python3
"""
execute_trades.py - 执行实际的股票交易
change everytime when i want to execute trades
"""

import sys
import os
sys.path.append('.')
from alpaca import AlpacaConnection
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def execute_half_trades():
    """执行一半数量的交易"""
    
    # 原始交易计划
    original_trades = [
        {'symbol': 'AAPL', 'qty': 941},
        {'symbol': 'MSFT', 'qty': 398},
        {'symbol': 'GOOGL', 'qty': 1125},
        {'symbol': 'AMZN', 'qty': 899},
        {'symbol': 'TSLA', 'qty': 645}
    ]
    
    # 计算一半的数量
    half_trades = []
    for trade in original_trades:
        half_qty = trade['qty'] // 2
        half_trades.append({
            'symbol': trade['symbol'],
            'qty': half_qty
        })
    
    logger.info("🚀 开始执行交易...")
    logger.info(f"💰 总共要执行 {len(half_trades)} 笔交易")
    
    # 创建Alpaca连接
    try:
        alpaca = AlpacaConnection()
        logger.info("✅ Alpaca连接建立成功")
    except Exception as e:
        logger.error(f"❌ 无法连接到Alpaca: {e}")
        return
    
    # 获取账户信息
    account_info = alpaca.get_account_info()
    if not account_info:
        logger.error("❌ 无法获取账户信息")
        return
    
    logger.info(f"💰 账户信息:")
    logger.info(f"   总权益: ${account_info['equity']:,.2f}")
    logger.info(f"   可用现金: ${account_info['cash']:,.2f}")
    logger.info(f"   买入力: ${account_info['buying_power']:,.2f}")
    
    # 获取当前价格并计算总投资金额
    logger.info("\n📊 获取当前股价...")
    current_prices = {}
    total_investment = 0
    
    for trade in half_trades:
        symbol = trade['symbol']
        qty = trade['qty']
        
        try:
            # 获取当前价格
            if alpaca.api is not None:
                bars = alpaca.api.get_bars(symbol, '1Day', limit=1).df
                if not bars.empty:
                    current_price = float(bars['close'].iloc[-1])
                    current_prices[symbol] = current_price
                    estimated_value = current_price * qty
                    total_investment += estimated_value
                    
                    logger.info(f"   {symbol}: ${current_price:.2f} × {qty} = ${estimated_value:,.2f}")
                else:
                    logger.warning(f"   ⚠️  无法获取 {symbol} 的价格")
            else:
                logger.error(f"   ❌ API连接为空，无法获取 {symbol} 的价格")
                
        except Exception as e:
            logger.error(f"   ❌ 获取 {symbol} 价格失败: {e}")
    
    logger.info(f"\n💵 预计总投资金额: ${total_investment:,.2f}")
    
    # 检查是否有足够的资金
    if total_investment > account_info['buying_power']:
        logger.error(f"❌ 资金不足! 需要 ${total_investment:,.2f}，但只有 ${account_info['buying_power']:,.2f}")
        return
    
    # 确认执行
    logger.info("\n⚠️  即将执行实际交易!")
    logger.info("📋 交易明细:")
    for trade in half_trades:
        symbol = trade['symbol']
        qty = trade['qty']
        if symbol in current_prices:
            price = current_prices[symbol]
            value = price * qty
            logger.info(f"   买入 {qty} 股 {symbol} @ ${price:.2f} = ${value:,.2f}")
    
    # 最后确认
    response = input("\n🤔 确认执行这些交易? (输入 'YES' 确认, 其他任何输入取消): ")
    if response.upper() != 'YES':
        logger.info("❌ 交易已取消")
        return
    
    # 执行交易
    logger.info("\n🎯 开始执行交易...")
    executed_orders = []
    
    for trade in half_trades:
        symbol = trade['symbol']
        qty = trade['qty']
        
        if symbol not in current_prices:
            logger.warning(f"⚠️  跳过 {symbol} - 无法获取价格")
            continue
        
        try:
            logger.info(f"📈 正在买入 {qty} 股 {symbol}...")
            
            # 执行市价买单
            order = alpaca.place_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                order_type='market',
                time_in_force='day'
            )
            
            if order:
                executed_orders.append({
                    'symbol': symbol,
                    'qty': qty,
                    'order_id': order.id,
                    'status': order.status,
                    'estimated_price': current_prices[symbol]
                })
                
                logger.info(f"✅ {symbol} 订单提交成功!")
                logger.info(f"   订单ID: {order.id}")
                logger.info(f"   状态: {order.status}")
                
                # 短暂延迟避免过于频繁的请求
                time.sleep(2)
                
            else:
                logger.error(f"❌ {symbol} 订单提交失败")
                
        except Exception as e:
            logger.error(f"❌ 执行 {symbol} 交易失败: {e}")
    
    # 交易总结
    logger.info("\n📊 交易执行总结:")
    logger.info(f"✅ 成功提交 {len(executed_orders)} 笔订单")
    
    if executed_orders:
        logger.info("\n📋 已提交的订单:")
        total_executed_value = 0
        for order in executed_orders:
            estimated_value = order['qty'] * order['estimated_price']
            total_executed_value += estimated_value
            logger.info(f"   {order['symbol']}: {order['qty']} 股 @ ~${order['estimated_price']:.2f} = ~${estimated_value:,.2f}")
            logger.info(f"     订单ID: {order['order_id']}")
        
        logger.info(f"\n💰 总投资金额: ~${total_executed_value:,.2f}")
        logger.info("\n⏰ 注意: 这些是市价单，实际成交价格可能与估计价格有所不同")
        logger.info("🔍 请通过Alpaca平台或API查看订单的实际执行情况")
    
    # 查看最新的持仓
    logger.info("\n📈 更新后的持仓:")
    time.sleep(5)  # 等待订单处理
    
    try:
        positions = alpaca.get_positions()
        if not positions.empty:
            for _, pos in positions.iterrows():
                logger.info(f"   {pos['symbol']}: {pos['qty']} 股 = ${pos['market_value']:,.2f}")
        else:
            logger.info("   暂无持仓或订单还在处理中")
    except Exception as e:
        logger.error(f"获取持仓失败: {e}")

if __name__ == "__main__":
    execute_half_trades() 