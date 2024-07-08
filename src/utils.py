import logging
import datetime as dt
import pandas as pd
import numpy as np

def check_negative_balance(series):
    if (series <= 0).any():
        # Find the first index where the value is 0 or negative
        first_zero_or_negative = series[series <= 0].index[0]

        # Set all values from that index onward to 0
        series[first_zero_or_negative:] = 0

        return False
    
    # No negative Balance:
    return True

def get_net_balance(balance,reb_cost:float):
    if balance > 0:
        net_balance = round(balance * (1-reb_cost),2)
    else:
        net_balance = 0
    
    return net_balance


def port_balance_calc(portfolio_weights: pd.Series, balance: float, date_filtered_returns: pd.DataFrame,kd:float) -> pd.Series:
    """
    Simulates a portfolio position balance variation over time.

    Parameters:
    portfolio_weights (pd.Series): The weights of each asset in the portfolio.
    balance (float): The total balance of the portfolio.
    date_filtered_returns (pd.DataFrame): DataFrame of the returns of each asset over time.

    Returns:
    pd.Series:  balance DataFrame and the last portfolio weights.
    """
    try:
        # If portfolio weights compatible:
        if len(portfolio_weights) != portfolio_weights.isna().sum():
            
            # Calculate the value of each asset in the portfolio
            initial_portfolio = portfolio_weights * balance

            # Calculate cumulative returns over time
            cumulative_log_returns = np.log(1 + date_filtered_returns).cumsum()

            # Calculate the Net Asset Value (NAV) of each asset over time
            position_NAV = (initial_portfolio * np.exp(cumulative_log_returns)).round(2)

            # Calculate the total balance over time
            balance_over_time = position_NAV.sum(axis=1)

            # Check for negative balance:
            neg_check = check_negative_balance(series=balance_over_time)

            # Apply cost of leverage to balance over time:
            if kd > 0 and neg_check:
                balance_net_pct = balance_over_time.pct_change() - kd
                log_net_pct = np.log(1+balance_net_pct)
                balance_over_time = balance * np.exp(log_net_pct.cumsum()) 

            # Calculate the last balance
            final_balance = balance_over_time.iloc[-1]

            # Calculate the weights of the last portfolio
            final_portfolio = position_NAV.iloc[-1] / final_balance
        
        else: # If model weights all nan:

            # Add transaction costs of selling/buying
            dates_index = date_filtered_returns.index
            balance_over_time = pd.Series(balance, index=dates_index)
            final_portfolio = pd.Series(0,index=portfolio_weights.index)

        return balance_over_time, final_portfolio
    
    except Exception as e:
        logging.exception(f'ERROR Calculating BT Balance | {e}')

def get_balance(starting_balance,rebalance_dates,portfolios,end_date,sim_price_data,trans_cost,annual_kd):
    
    # Initialize Object to store balance data:
    all_dates_balance = {}

    # Set starting balance:
    balance = starting_balance

    # Standarize end_date:
    try:
        end_date = end_date.date()
    except:
        pass

    # Get rebalance dates from portfolios:
    returns_start_date = rebalance_dates[0]

    # Used to calculate rebalance costs:
    last_weights = pd.Series(0,index=sim_price_data.columns)
    
    # Calculate Strategy balance:
    n_rebalances = len(rebalance_dates)
    for reb_n in range(0,n_rebalances):

        portfolio_weights = portfolios.iloc[:,reb_n].dropna()

        # Set Date to with next portfolio rebalance
        try:
            if reb_n+1 < n_rebalances:
                next_port = portfolios.iloc[:,reb_n+1]
                date_to = next_port.name.date()
            else:
                # Check End of simulation
                date_to = end_date

                if returns_start_date == date_to:
                    break
        
        except Exception as e:
            logging.exception(f'ERROR setting BT dates | {e}')
        
        # Filter Simulation Price df to match portfolio rebalance range & assets:
        portfolio_weights = portfolio_weights[portfolio_weights!=0]
        date_filtered_returns = sim_price_data[portfolio_weights.index].loc[returns_start_date : date_to].pct_change(fill_method=None).dropna()
        
        # Get costs:
        reb_cost, lev_cost = get_costs(trans_cost=trans_cost,portfolio_weights=portfolio_weights,
                                       last_weights=last_weights,annual_kd=annual_kd)

        # Apply Transaction costs:
        net_balance = get_net_balance(balance=balance,reb_cost=reb_cost)

        # Balance period calculation:
        balance_df,last_weights = port_balance_calc(portfolio_weights = portfolio_weights, balance=net_balance, 
                                               date_filtered_returns = date_filtered_returns, kd=lev_cost)

        # Final Balance:
        balance = balance_df.iloc[-1]
        all_dates_balance[reb_n] = balance_df
        
        # Set date & balance for next sim: 
        returns_start_date = date_to
    
    return all_dates_balance

def calculate_rebalance_cost(current_portfolio: pd.Series, prev_portfolio: pd.Series, transaction_cost: float) -> float:
    """
    Calculates the net transaction cost of rebalancing a portfolio.

    Parameters:
    current_portfolio (pd.Series): The weights of each asset in the current portfolio.
    prev_portfolio (pd.Series): The weights of each asset in the last portfolio.
    transaction_cost (float): The transaction cost per unit weight transferred.

    Returns:
    float: The net transaction cost of rebalancing the portfolio.
    """
    try:
        # Align both portfolios to ensure matching indices
        current_portfolio, prev_portfolio = current_portfolio.align(prev_portfolio, fill_value=0)

        # Calculate the absolute difference in weights between the two portfolios
        weight_difference = abs(current_portfolio - prev_portfolio)

        # Calculate the total transaction cost
        reb_cost = (weight_difference * transaction_cost).sum().round(6)

        return reb_cost
    
    except Exception as e:
        logging.exception(f'ERROR calculating rebalance costs | {e}')

def calculate_leverage_costs(annual_cost_of_debt, portfolio_weights, days_in_year=360):
    """
    Calculate the leverage costs of rebalancing and leverage for a portfolio.

    Parameters:
    annual_cost_of_debt (float): The annual cost of debt as a percentage.
    portfolio_weights (pd.Series): The current portfolio weights.
    days_in_year (int, optional): The number of days in a year for interest calculation. Defaults to 360.

    Returns:
    float: The total costs incurred from rebalancing and leverage.
    """
    try:
        leverage_cost = 0
        if annual_cost_of_debt > 0:
            total_debt = abs(portfolio_weights[portfolio_weights<0].sum())
            if total_debt > 0:
                daily_cost_of_debt = annual_cost_of_debt / days_in_year
                leverage_cost = daily_cost_of_debt * total_debt

        return leverage_cost
    except Exception as e:
        logging.exception(f'ERROR calculating leverage costs | {e}')

def get_costs(trans_cost,portfolio_weights,last_weights,annual_kd):
    
    # Calc transaction cost:
    if trans_cost <= 0:
        reb_cost = 0
    else:
        reb_cost = calculate_rebalance_cost(current_portfolio=portfolio_weights,
                                            prev_portfolio=last_weights,
                                            transaction_cost=trans_cost)
        
    # Calculate leverage costs:
    lev_cost = calculate_leverage_costs(annual_cost_of_debt=annual_kd,
                                        portfolio_weights=portfolio_weights)
    
    return reb_cost,lev_cost