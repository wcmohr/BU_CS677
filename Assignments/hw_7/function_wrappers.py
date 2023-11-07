from functions import *
def trading_strategies(df,year, staggered_dict, predictions):
    """Args: df (stock with year, open, adj close) , year (year for which returns will be caluculated), staggered_dict (year-week:year-(week-1)), predictions (list [0,1,0,...,0,1,1,0] where 1s indicate predicted good weeks, on which trading will occur)"""
    weekly_returns = trade_from_close_close(df,staggered_dict)
    strategy_portfolio =  trade_labels(weekly_returns, year, predictions)
    buy_hold_portfolio =  buy_and_hold(weekly_returns,year)
    return strategy_portfolio, buy_hold_portfolio