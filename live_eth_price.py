import requests

# Function to get live ETH price from CoinGecko
def get_live_eth_price_usd():
    url = 'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd'
    response = requests.get(url)
    data = response.json()
    return data.get('ethereum', {}).get('usd', None)
