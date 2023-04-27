import papermill as pm
import logging


stocks = ['AAPL', 'OXY', 'GS', 'MCD', 'MSFT', 'KHC', 'PEP']

for stock in stocks:
    try:
        pm.execute_notebook(
           'stock_analysis/template.ipynb',
           f'stock_analysis/{stock}_analysis.ipynb',
           parameters = dict(ticker = stock, using_papermill = True)
        )
    except Exception as e:
        logging.error(f"There was an error with stock {stock} - {e}")
        
        
