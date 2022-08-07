import requests
import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import pypfopt

minPct = 0 #minimum percentage in each asset 
maxPct = 0.2 #maximum percentage in each asset 

exchange = ccxt.binance()
headers = {
    'X-CMC_PRO_API_KEY': '64562e37-ef2e-4bd9-9357-b55efa1ae423'
}
res = [i['symbol'] for i in requests.get('https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest', headers=headers).json()['data'] if 'stablecoin' not in i['tags']]
data={}
for i in res:
    try: 
        req = exchange.fetch_ohlcv(f'{i}USDT', '1d')
        returns = [x[4] for x in req][-365:]
        if len(returns) == 365:
            data[i] = returns
    except Exception as e:
        continue
df = pd.DataFrame(data)
mu = pypfopt.expected_returns.mean_historical_return(df)
S = pypfopt.risk_models.sample_cov(df)
ef = pypfopt.efficient_frontier.EfficientFrontier(mu, S, weight_bounds=(minPct, maxPct))
weights = ef.max_sharpe()
weightsDict = {key: value for (key, value) in dict(weights).items() if value > 0}
print(ef.portfolio_performance(verbose=True))
print(weightsDict)
labels = []
sizes = []
sum = []
newdf = pd.DataFrame()
for x, y in weightsDict.items():
    newdf[x] = df[x].pct_change()*y
    labels.append(f'{x}\n{round(y, 2)*100} %')
    sizes.append(y)
newdf['sum'] = newdf.sum(axis = 1)
print(newdf.to_string())

fig, ax = plt.subplots(2)
ax[0].plot(newdf['sum'].cumsum()*100)
ax[1].pie(sizes, labels=labels)
plt.show()
