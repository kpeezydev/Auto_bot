import ccxt
import os
import sys

# --- Configuration ---
EXCHANGE_ID = 'pionex'
API_KEY_ENV_VAR = 'PIONEX_API_KEY'
SECRET_ENV_VAR = 'PIONEX_SECRET'
TEST_SYMBOL = 'AVAX/USDT' # Example symbol, change if needed

# --- Get API Keys from Environment Variables ---
api_key = os.getenv(API_KEY_ENV_VAR)
secret = os.getenv(SECRET_ENV_VAR)

if not api_key or not secret:
    print(f"Error: Please set the environment variables {API_KEY_ENV_VAR} and {SECRET_ENV_VAR}")
    sys.exit(1) # Exit if keys are not found

# --- Initialize Exchange ---
print(f"Initializing {EXCHANGE_ID} exchange...")
try:
    exchange = ccxt.pionex({
        'apiKey': api_key,
        'secret': secret,
        # 'enableRateLimit': True, # Optional: Enable built-in rate limiting
        # Add any other Pionex specific options if needed
    })
    # You might need to load markets depending on the testnet/sandbox environment
    # exchange.load_markets()
    print(f"{EXCHANGE_ID} initialized successfully.")
except ccxt.AuthenticationError as e:
    print(f"Authentication Error: {e}. Check your API keys and permissions.")
    sys.exit(1)
except ccxt.ExchangeError as e:
    print(f"Exchange Error during initialization: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during initialization: {e}")
    sys.exit(1)

# --- Test 1: Fetch Balance ---
print("\nAttempting to fetch balance...")
try:
    balance = exchange.fetch_balance()
    print("Balance fetched successfully:")
    # Print specific balances if needed, e.g., USDT
    if 'USDT' in balance['total']:
        print(f"  Total USDT: {balance['total']['USDT']}")
    else:
        print("  USDT balance not found.")
    # print(balance) # Uncomment for full balance details
except ccxt.AuthenticationError as e:
    print(f"Authentication Error fetching balance: {e}. Check API key permissions.")
except ccxt.ExchangeNotAvailable as e:
    print(f"Exchange Not Available fetching balance: {e}")
except ccxt.NetworkError as e:
    print(f"Network Error fetching balance: {e}")
except ccxt.ExchangeError as e:
    print(f"Exchange Error fetching balance: {e}")
except Exception as e:
    print(f"An unexpected error occurred fetching balance: {e}")

# --- Test 2: Fetch Ticker ---
print(f"\nAttempting to fetch ticker for {TEST_SYMBOL}...")
try:
    ticker = exchange.fetch_ticker(TEST_SYMBOL)
    print(f"Ticker for {TEST_SYMBOL} fetched successfully:")
    print(f"  Last Price: {ticker.get('last')}")
    print(f"  Bid: {ticker.get('bid')}")
    print(f"  Ask: {ticker.get('ask')}")
    # print(ticker) # Uncomment for full ticker details
except ccxt.BadSymbol as e:
    print(f"Error fetching ticker: {e}. Symbol '{TEST_SYMBOL}' might be invalid for {EXCHANGE_ID}.")
except ccxt.ExchangeNotAvailable as e:
    print(f"Exchange Not Available fetching ticker: {e}")
except ccxt.NetworkError as e:
    print(f"Network Error fetching ticker: {e}")
except ccxt.ExchangeError as e:
    print(f"Exchange Error fetching ticker: {e}")
except Exception as e:
    print(f"An unexpected error occurred fetching ticker: {e}")

print("\nConnection test finished.")