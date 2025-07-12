import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlpacaConnection:
    """
    Alpaca API连接类，用于连接Alpaca交易平台
    基于提供的API凭据建立安全连接
    """
    
    def __init__(self):
        # API credentials from the image
        self.api_key = "PKIP2RHV9HG4SIBDG2OH"
        self.secret_key = "XOYSrS5YzPk9Td79dnwbM34FB2PCCQ9CgaWlMmZZ"
        # Fix the base URL - remove the duplicate /v2
        self.base_url = "https://paper-api.alpaca.markets"
        
        # Initialize API connection
        self.api = None
        self.connect()
    
    def connect(self):
        """
        建立与Alpaca API的连接
        """
        try:
            # Debug: Print the base URL being used
            logger.info(f"Attempting to connect to: {self.base_url}")
            
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            logger.info(f"Successfully connected to Alpaca API")
            logger.info(f"Account Status: {account.status}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            logger.error(f"Base URL used: {self.base_url}")
            logger.error(f"API Key: {self.api_key[:8]}...")  # Only show first 8 chars for security
            raise
    
    def get_account_info(self):
        """
        获取账户信息
        """
        try:
            if self.api is None:
                logger.error("API connection is None")
                return None
                
            account = self.api.get_account()
            return {
                'account_id': account.id,
                'status': account.status,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_positions(self):
        """
        获取当前持仓信息
        """
        try:
            if self.api is None:
                logger.error("API connection is None")
                return pd.DataFrame()
                
            positions = self.api.list_positions()
            position_data = []
            
            for position in positions:
                position_data.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc)
                })
            
            return pd.DataFrame(position_data)
        
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return pd.DataFrame()
    
    def get_all_tradable_stocks(self):
        """
        获取所有可交易股票列表
        """
        try:
            if self.api is None:
                logger.error("API connection is None")
                return []
                
            # 获取所有可交易的美股
            assets = self.api.list_assets(
                status='active',
                asset_class='us_equity'
            )
            
            # 筛选条件：可交易、主要交易所、非仙股
            major_exchanges = ['NASDAQ', 'NYSE', 'NYSEARCA']
            tradable_stocks = []
            
            for asset in assets:
                if (asset.tradable and 
                    asset.exchange in major_exchanges and
                    asset.marginable):  # 可融资的股票通常质量更好
                    tradable_stocks.append(asset.symbol)
            
            logger.info(f"Found {len(tradable_stocks)} tradable stocks")
            return tradable_stocks
            
        except Exception as e:
            logger.error(f"Error getting tradable stocks: {e}")
            return []
    
    def get_current_prices(self, symbols):
        """获取股票当前价格"""
        try:
            if self.api is None:
                logger.error("API connection is None")
                return {}
                
            current_prices = {}
            for symbol in symbols:
                try:
                    bars = self.api.get_bars(symbol, '1Day', limit=1).df
                    if not bars.empty:
                        current_prices[symbol] = float(bars['close'].iloc[-1])
                except:
                    continue  # 跳过无法获取价格的股票
                    
            return current_prices
        except Exception as e:
            logger.error(f"Failed to get prices: {e}")
            return {}
    
    def place_order(self, symbol, qty, side='buy', order_type='market', time_in_force='day'):
        """
        下单交易
        """
        try:
            if self.api is None:
                logger.error("API connection is None")
                return None
                
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            
            logger.info(f"Order placed: {side} {qty} shares of {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None


def create_alpaca_connection():
    """
    创建Alpaca连接实例
    """
    try:
        return AlpacaConnection()
    except Exception as e:
        logger.error(f"Failed to create Alpaca connection: {e}")
        return None


# 如果直接运行此文件，执行简单测试
if __name__ == "__main__":
    # 测试连接
    alpaca = create_alpaca_connection()
    
    if alpaca is None:
        logger.error("Failed to establish Alpaca connection. Exiting...")
        exit(1)
    
    # 获取账户信息
    account_info = alpaca.get_account_info()
    print("Account Info:", account_info)
    
    # 获取所有可交易股票
    print("\n获取所有可交易股票...")
    all_stocks = alpaca.get_all_tradable_stocks()
    print(f"总共找到 {len(all_stocks)} 只可交易股票")
    print(f"前20只股票: {all_stocks[:20]}")