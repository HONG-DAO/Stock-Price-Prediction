import numpy as np
import arch
import warnings
import json
import os
import pickle

class AutomaticGARCHPredictor:
    def __init__(self, max_p=3, max_q=3, output_dir="paras/Statistic"):
        """
        Initialize the GARCH predictor with flexible parameter search
        
        Parameters:
        -----------
        max_p : int, optional (default=3)
            Maximum number of ARCH terms to consider
        max_q : int, optional (default=3)
            Maximum number of GARCH terms to consider
        output_dir : str, optional (default="params")
            Directory to store parameter files
        """
        self.max_p = max_p
        self.max_q = max_q
        self.best_model = None
        self.best_params = None
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def _calculate_returns(self, prices):
        """
        Calculate log returns from price series
        
        Parameters:
        -----------
        prices : array-like
            Stock price time series
        
        Returns:
        --------
        array
            Log returns
        """
        prices = np.array(list(prices))
        return np.log(prices[1:] / prices[:-1])
    
    def _find_best_garch_specification(self, returns):
        """
        Automatically find the best GARCH model specification
        
        Parameters:
        -----------
        returns : array-like
            Log returns of the stock price
        
        Returns:
        --------
        tuple
            Best (p, q) parameters and corresponding model
        """
        best_aic = np.inf
        best_specification = (1, 1)
        best_model = None
        
        warnings.filterwarnings('ignore')
        
        for p in range(1, self.max_p + 1):
            for q in range(1, self.max_q + 1):
                try:
                    model = arch.arch_model(returns, p=p, q=q, 
                                            mean='Zero', vol='Garch', 
                                            dist='Normal')
                    results = model.fit(disp='off')
                    
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_specification = (p, q)
                        best_model = results
                except Exception:
                    continue
        
        return best_specification, best_model
    
    def fit(self, prices, stock_symbol, session):
        """
        Fit GARCH model to price data and save parameters to a file
        
        Parameters:
        -----------
        prices : array-like
            Stock price time series
        stock_symbol : str
            Stock symbol (e.g., 'AAPL')
        session : str
            Session identifier (e.g., '1d', '30m')
        
        Returns:
        --------
        self : object
            Fitted predictor
        """
        returns = self._calculate_returns(prices)
        (p, q), model = self._find_best_garch_specification(returns)
        
        self.best_params = {
            'p': p,
            'q': q,
            'omega': model.params['omega'],
            'alpha': model.params.get(f'alpha[{p}]', None),
            'beta': model.params.get(f'beta[{q}]', None)
        }
        self.best_model = model
        
        # Save params
        params_filename = os.path.join(self.output_dir, f"params_{stock_symbol}_{session}.json")
        with open(params_filename, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        # Save model
        model_filename = os.path.join(self.output_dir, f"model_{stock_symbol}_{session}.pkl")
        with open(model_filename, 'wb') as f:
            pickle.dump(self.best_model, f)

        return self
    
    def load_params(self, stock_symbol, session):
        """
        Load GARCH parameters from a file
        
        Parameters:
        -----------
        stock_symbol : str
            Stock symbol (e.g., 'AAPL')
        session : str
            Session identifier (e.g., '1d', '30m')
        
        Returns:
        --------
        dict
            Loaded parameters
        """
        params_filename = os.path.join(self.output_dir, f"params_{stock_symbol}_{session}.json")
        model_filename = os.path.join(self.output_dir, f"model_{stock_symbol}_{session}.pkl")
        
        if os.path.exists(params_filename):
            with open(params_filename, 'r') as f:
                self.best_params = json.load(f)

            with open(model_filename, 'rb') as f:
                self.best_model = pickle.load(f)

            return self.best_params, self.best_model
        else:
            raise FileNotFoundError(f"Parameter file for {stock_symbol} {session} not found.")

    def predict_next_point(self, prices, stock_symbol, session, forecast_horizon=1):
        """
        Predict next stock price point using saved parameters
        
        Parameters:
        -----------
        prices : array-like
            Historical stock price time series
        stock_symbol : str
            Stock symbol (e.g., 'AAPL')
        session : str
            Session identifier (e.g., '1d', '30m')
        forecast_horizon : int, optional (default=1)
            Number of steps to forecast ahead
        
        Returns:
        --------
        float
            Predicted next price point
        """
        try:
            self.load_params(stock_symbol, session)
        except FileNotFoundError:
            self.fit(prices, stock_symbol, session)

        returns = self._calculate_returns(prices)
        forecast = self.best_model.forecast(horizon=forecast_horizon)
        
        last_price = prices[-1]
        mean_return = np.mean(returns)
        predicted_price = last_price * np.exp(mean_return * np.mean(forecast.residual_variance))
        
        return predicted_price

    @staticmethod
    def run_daily_task(stock_symbols, sessions, price_data):
        """
        Run daily task to find and save best GARCH parameters for all stocks and sessions
        
        Parameters:
        -----------
        stock_symbols : list
            List of stock symbols
        sessions : list
            List of session identifiers
        price_data : dict
            Dictionary containing price series for each stock and session
        """
        for stock_symbol in stock_symbols:
            for session in sessions:
                if (stock_symbol in price_data) and (session in price_data[stock_symbol]):
                    prices = price_data[stock_symbol][session]
                    predictor = AutomaticGARCHPredictor()
                    predictor.fit(prices, stock_symbol, session)

if __name__ == "__main__":
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'BRK-B', 'META', 'JNJ', 'XOM']
    sessions = ['1d', '30m', '15m', '5m', '1m', 'tick']
    
    # Example price data (to be replaced with actual data)
    price_data = {symbol: {session: np.random.rand(100) * 100 for session in sessions} for symbol in stock_symbols}

    AutomaticGARCHPredictor.run_daily_task(stock_symbols, sessions, price_data)
