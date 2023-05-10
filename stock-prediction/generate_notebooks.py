import papermill as pm
import logging


stocks = ['AAPL', 'OXY', 'GS', 'MCD', 'MSFT', 'KHC', 'PEP']

RUN_STOCK_ANALYSIS = False
RUN_CLASSIFIER = True

for stock in stocks:
    try:
        if RUN_STOCK_ANALYSIS:
            print("Generating notebooks for stock analysis")
            pm.execute_notebook(
               'notebooks/template.ipynb',
               f'notebooks/{stock}_analysis.ipynb',
               parameters = dict(ticker = stock, using_papermill = True)
            )
        if RUN_CLASSIFIER:
            print("Generating notebooks for classifier")
            pm.execute_notebook(
               'notebooks/classifier.ipynb',
               f'notebooks/{stock}_classifier.ipynb',
               parameters = dict(ticker = stock, using_papermill = True)
            )
    except Exception as e:
        logging.error(f"There was an error with stock {stock} - {e}")
        
        
