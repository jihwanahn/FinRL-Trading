

# FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading
<p align="center">
  <img src="https://github.com/user-attachments/assets/80fe89bb-fb09-4267-b29a-76030512f8cf" width="500">
</p>

[![Downloads](https://static.pepy.tech/badge/finrl-trading)](https://pepy.tech/project/finrl-trading)
[![Downloads](https://static.pepy.tech/badge/finrl-trading/week)](https://pepy.tech/project/finrl-trading)
[![Join Discord](https://img.shields.io/badge/Discord-Join-blue)](https://discord.gg/trsr8SXpW5)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/finrl-trading.svg)](https://pypi.org/project/finrl-trading/)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/FinRL-Trading.svg?color=brightgreen)
![](https://img.shields.io/github/issues-raw/AI4Finance-Foundation/FinRL-Trading?label=Issues)
![](https://img.shields.io/github/issues-closed-raw/AI4Finance-Foundation/FinRL-Trading?label=Closed+Issues)
![](https://img.shields.io/github/issues-pr-raw/AI4Finance-Foundation/FinRL-Trading?label=Open+PRs)
![](https://img.shields.io/github/issues-pr-closed-raw/AI4Finance-Foundation/FinRL-Trading?label=Closed+PRs)


![Visitors](https://api.visitorbadge.io/api/VisitorHit?user=AI4Finance-Foundation&repo=FinRL-Trading&countColor=%23B17A)
[![](https://dcbadge.limes.pink/api/server/trsr8SXpW5?cb=1)](https://discord.gg/trsr8SXpW5)

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv_2603.21330-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2603.21330)

---

## 📖 About

**FinRL-X** is a next-generation, **AI-native** quantitative trading infrastructure that redefines how researchers and practitioners build, test, and deploy algorithmic trading strategies. Introduced in our paper *"FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading"* ([arXiv:2603.21330](https://arxiv.org/abs/2603.21330)), FinRL-X succeeds the original [FinRL](https://github.com/AI4Finance-Foundation/FinRL) framework with a fully modernized architecture designed for the LLM and agentic AI era.

> FinRL-X is **not just a library** — it is a full-stack trading platform engineered around modularity, reproducibility, and production-readiness, supporting everything from ML-based stock selection and professional backtesting to live brokerage execution.
---

## 🌟 From FinRL to FinRL-X: The Full Evolution

FinRL-X is the **Stage 3.0 production release** of the AI4Finance development roadmap — built for institutions and practitioners who need reliability, modularity, and AI-native capabilities beyond what the original FinRL framework provides.

| Stage | Maturity | Target Users | Project | Core Capability |
|:---:|:---:|---|---|---|
| **0.0** | Entry | Practitioners | [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta) | Gym-style market environments & benchmarks |
| **1.0** | Proof-of-Concept | Developers | [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | Automatic train → test → trade pipeline |
| **2.0** | Professional | Researchers | [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) | State-of-the-art DRL algorithms |
| **3.0** 🚀 | **Production** | **Institutions & Quants** | **FinRL-X ← you are here** | **AI-native, modular, live-trading infrastructure** |

> **FinRL** (2020) proved that deep reinforcement learning could automate trading. **FinRL-X** (2025) makes it production-ready — with cleaner architecture, smarter data pipelines, professional backtesting, and live brokerage integration, all redesigned for the LLM and agentic AI era.

---

---

## 🔄 FinRL-X vs. FinRL: What Changed

| Capability | FinRL (Stage 1.0) | FinRL-X (Stage 3.0) |
|---|---|---|
| **Paradigm** | Deep Reinforcement Learning | AI-Native (ML + DRL + LLM-ready) |
| **Architecture** | Three-layer coupled monolith | Fully decoupled modular layers |
| **Strategies** | DRL agents (A2C, DDPG, PPO, SAC, TD3) | ML selection + DRL timing + extensible base |
| **Data Layer** | 14 manually-wired processors | Auto-select: Yahoo Finance → FMP → WRDS |
| **Backtesting** | Custom hand-rolled evaluation loops | Professional `bt` library engine |
| **Live Trading** | Basic Alpaca support | Full multi-account integration + risk controls |
| **Configuration** | `config.py` + `config_tickers.py` | Type-safe Pydantic + `.env` multi-env |
| **Risk Management** | Gym environment constraints only | Order · portfolio · strategy-level controls |
| **Target Users** | Researchers & students | Quants, institutions, production deployments |
| **Paper** | [arXiv:2011.09607](https://arxiv.org/abs/2011.09607) | [arXiv:2603.21330](https://arxiv.org/abs/2603.21330) |

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **AI-Native Strategy Framework** | Pluggable strategies including ML stock selection, DRL agents, and LLM-driven signal generation |
| 📈 **Risk Management** | Comprehensive risk controls: position limits, turnover caps, and drawdown guards |
| 💰 **Live Trading** | Alpaca brokerage integration with paper and live trading modes |
| 🔧 **Modular Architecture** | Clean, extensible design following software engineering best practices |
| 🗄️ **Multi-Source Data** | Yahoo Finance · FMP · WRDS — intelligent source selection with SQLite caching |
| 📊 **Professional Backtesting** | Powered by the `bt` library with benchmark comparison and transaction cost simulation |
| ⚙️ **Type-Safe Configuration** | Pydantic-based settings with environment variable support across dev/test/prod |

---

## 🏗️ Architecture

<div align="center">
  <img src="https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master/figs/FinRL_X_Framework.png" width="900"/>
  <br/><em>FinRL-X layered architecture: Data → Strategy → Backtest → Live Trading</em>
</div>

<br/>

```
finrl-trading/
├── src/
│   ├── config/                     # ⚙️  Centralized configuration management
│   │   └── settings.py             #     Pydantic-based settings + environment variables
│   ├── data/                       # 🗄️  Data acquisition and processing
│   │   ├── data_fetcher.py         #     Multi-source integration (Yahoo / FMP / WRDS)
│   │   ├── data_processor.py       #     Feature engineering & data cleaning
│   │   └── data_store.py           #     SQLite persistence with caching
│   ├── backtest/                   # 📊  Backtesting engine
│   │   └── backtest_engine.py      #     bt-powered engine with benchmark comparison
│   ├── strategies/                 # 🤖  Trading strategies
│   │   ├── base_strategy.py        #     Abstract strategy framework
│   │   └── ml_strategy.py          #     Random Forest stock selection
│   ├── trading/                    # 💰  Live trading execution
│   │   ├── alpaca_manager.py       #     Alpaca API integration (multi-account)
│   │   ├── trade_executor.py       #     Order management & risk controls
│   │   └── performance_analyzer.py #     Real-time P&L tracking
│   └── main.py                     # 🚀  CLI entry point
├── examples/
│   ├── FinRL_Full_Workflow.ipynb   # 📓  Complete workflow tutorial (start here!)
│   └── README.md
├── data/                           # Runtime data storage (gitignored)
├── logs/                           # Application logs (gitignored)
├── requirements.txt
└── setup.py
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Alpaca Account** — for live / paper trading ([sign up free](https://alpaca.markets/))
- **FMP API Key** — optional; Yahoo Finance works as a free default

### Install via PyPI

```bash
pip install finrl-trading
```

### Install from Source

```bash
# 1. Clone
git clone https://github.com/AI4Finance-Foundation/FinRL-Trading.git
cd FinRL-Trading

# 2. Install
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
nano .env    # Windows: notepad .env

# 4. Launch the full workflow tutorial
jupyter notebook examples/FinRL_Full_Workflow.ipynb
```

### Migration Guide: Coming from FinRL?

If you've been using FinRL's `train.py → test.py → trade.py` pipeline:

```
FinRL                               →   FinRL-X
──────────────────────────────────────────────────────────────────
finrl/meta/data_processor.py        →   src/data/data_fetcher.py
finrl/train.py                      →   strategy.generate_weights() + BacktestEngine
finrl/trade.py                      →   TradeExecutor.execute_portfolio_rebalance()
config.py + config_tickers.py       →   src/config/settings.py  (Pydantic + .env)
gym.Env subclassing                 →   BaseStrategy.generate_weights()  (no gym needed)
Manual DRL algorithm config         →   StrategyConfig  (declarative, type-checked)
```

---

## 🔑 Configuration

```bash
# ── Application ─────────────────────────────────────────────────
ENVIRONMENT=development          # development | staging | production
APP_NAME="FinRL-X Trading"

# ── Alpaca API (required for live/paper trading) ─────────────────
APCA_API_KEY=your_alpaca_api_key
APCA_API_SECRET=your_alpaca_api_secret
APCA_BASE_URL=https://paper-api.alpaca.markets   # Use live URL for real money

# ── Data Sources (priority: FMP > WRDS > Yahoo Finance) ──────────
FMP_API_KEY=your_fmp_api_key                     # Optional; Yahoo is the free default

# ── Risk Controls ─────────────────────────────────────────────────
TRADING_MAX_ORDER_VALUE=100000          # Max single order ($)
TRADING_MAX_PORTFOLIO_TURNOVER=0.5      # Max 50% portfolio turnover per rebalance
STRATEGY_MAX_WEIGHT_PER_STOCK=0.1       # Max 10% per position

# ── Caching ───────────────────────────────────────────────────────
DATA_CACHE_TTL_HOURS=24
DATA_MAX_CACHE_SIZE_MB=1000
```

---

## 📓 Tutorial: Full Workflow

```bash
jupyter notebook examples/FinRL_Full_Workflow.ipynb
```

| Step | Content |
|:---:|---|
| **1** | Build S&P 500 universe; download multi-source fundamental + price data |
| **2** | Engineer features for ML model training |
| **3** | Train and evaluate a Random Forest stock selection strategy |
| **4** | Run professional backtests vs. VOO / QQQ benchmarks |
| **5** | Execute via Alpaca paper trading; analyze live P&L |

---

## 📚 Usage Examples

### Data Acquisition

```python
from src.data.data_fetcher import get_data_manager

# Intelligently selects best available data source
manager = get_data_manager()

info = manager.get_source_info()
print(f"Active source: {info['current_source']}")
print(f"Available:     {info['available_sources']}")

tickers = ['AAPL', 'MSFT', 'GOOGL']

fundamentals = manager.get_fundamental_data(tickers, '2020-01-01', '2023-12-31')
prices        = manager.get_price_data(tickers,        '2020-01-01', '2023-12-31')
sp500         = manager.get_sp500_components()
```

> **vs. FinRL**: No more manually choosing between `YahooDownloader`, `AlpacaProcessor`, or `WRDSProcessor`. FinRL-X auto-detects available credentials and selects the optimal source automatically.

### Strategy Development

```python
from src.strategies.ml_strategy import MLStockSelectorStrategy
from src.strategies.base_strategy import StrategyConfig

config = StrategyConfig(
    name="ML Stock Selector",
    parameters={
        'model_type': 'random_forest',
        'top_n': 30,
        'sector_neutral': True   # Enforces sector diversification
    },
    risk_limits={'max_weight': 0.1}
)

strategy = MLStockSelectorStrategy(config)
result = strategy.generate_weights({'fundamentals': fundamentals, 'prices': prices})
print(result.weights.head())
```

**Add a custom strategy in minutes:**

```python
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult

class MyAlphaStrategy(BaseStrategy):
    """Drop-in for any built-in strategy — no gym.Env required."""
    def generate_weights(self, data, **kwargs) -> StrategyResult:
        # Your alpha logic here
        pass
```

> **vs. FinRL**: No need to define `state_space`, `action_space`, or reward functions. FinRL-X's `BaseStrategy` produces portfolio weights directly, cleanly separating alpha generation from execution.

### Backtesting

```python
from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=1_000_000,
    rebalance_freq='Q',           # Q=Quarterly, M=Monthly, W=Weekly
    transaction_cost=0.001,       # 10 bps round-trip
    benchmark_tickers=['VOO', 'QQQ', 'SPY']
)

engine = BacktestEngine(config)
result = engine.run_backtest(
    strategy_name="ML Stock Selector",
    weight_signals=ml_weights,
    price_data=prices
)

print(f"Total Return:      {result.metrics['total_return']:.2%}")
print(f"Annualized Return: {result.annualized_return:.2%}")
print(f"Sharpe Ratio:      {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown:      {result.metrics['max_drawdown']:.2%}")

engine.plot_results(result)   # Full performance dashboard
```

> **vs. FinRL**: FinRL uses hand-rolled evaluation loops. FinRL-X leverages the `bt` library, providing multi-benchmark comparison, rolling statistics, transaction cost simulation, and automated dashboards out of the box.

### Live Trading

```python
from src.trading.alpaca_manager import create_alpaca_account_from_env, AlpacaManager
from src.trading.trade_executor import TradeExecutor, ExecutionConfig

account  = create_alpaca_account_from_env()
alpaca   = AlpacaManager([account])     # Multi-account supported
executor = TradeExecutor(
    alpaca,
    ExecutionConfig(max_order_value=100_000, risk_checks_enabled=True)
)

target_weights = {'AAPL': 0.3, 'MSFT': 0.3, 'GOOGL': 0.4}
result = executor.execute_portfolio_rebalance(target_weights)

print(f"Orders placed:  {len(result.orders_placed)}")
print(f"Status:         {'✅ Success' if result.success else '❌ Failed'}")
```

---

## 🧩 Core Components

### 🗄️ Data Layer — `src/data/`

| Module | Role |
|---|---|
| `data_fetcher.py` | **Intelligent multi-source manager**: auto-selects Yahoo Finance → FMP → WRDS |
| `data_processor.py` | Feature engineering, technical indicators, factor construction, quality checks |
| `data_store.py` | SQLite persistence with TTL caching, deduplication, version tracking |

**Data source comparison** (vs. FinRL's 14 manually configured sources):

| Source | Asset Class | Access | Notes |
|---|---|---|---|
| Yahoo Finance | US Stocks, ETFs | Free | Default fallback; no API key needed |
| Financial Modeling Prep | US Securities | API Key | Premium fundamental + price data |
| WRDS / TAQ | US Equities | Academic | Intraday tick-level data |

### 🤖 Strategy Framework — `src/strategies/`

| Strategy | Type | Description |
|---|---|---|
| Equal Weight | Baseline | Uniform portfolio allocation |
| Market Cap Weighted | Baseline | Cap-weighted universe weighting |
| ML Stock Selection | ML | Random Forest factor model |
| Sector Neutral ML | ML | Sector-constrained selection |
| DRL Timing *(planned)* | DRL | PPO / SAC market timing agent |
| LLM Signal *(planned)* | LLM | FinGPT-powered sentiment alpha |

### 📊 Backtesting Engine — `src/backtest/`

Powered by [`bt`](https://github.com/pmorissette/bt):

- Multi-period, multi-benchmark comparison (any ticker)
- Configurable rebalance frequency: Q / M / W / custom
- Transaction cost and slippage simulation
- Rolling Sharpe, drawdown timelines, sector attribution
- HTML / PDF performance report generation

### 💰 Trading System — `src/trading/`

| Module | Responsibility |
|---|---|
| `alpaca_manager.py` | Multi-account Alpaca client; paper ↔ live switching |
| `trade_executor.py` | Order routing, pre-trade risk checks, order lifecycle tracking |
| `performance_analyzer.py` | Real-time position tracking, P&L attribution, exposure reporting |

---

## 📊 Performance Metrics
<p align="center">
  <img src="https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master/figs/All_Backtests_v2.png" width="1000">
</p>
<p align="center">
  <img src="https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master/figs/Paper_Trading.png" width="1000">
</p>

<p align="center">
  <img src="https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master/figs/Sector_Rotation_Standalone.png" width="1000">
</p>

<p align="center">
  <img src="https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master/figs/DRL_Timing_Backtest.png" width="1000">
</p>
The backtesting engine provides comprehensive quantitative analysis:

### Return Metrics
- **Total Return**: Cumulative portfolio performance
- **Annualized Return**: Time-weighted annual performance
- **Alpha**: Excess return over benchmark

### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted returns (Return ÷ Volatility)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Calmar Ratio**: Return ÷ Maximum Drawdown

### Tail Risk Measures
- **Skewness & Kurtosis**: Return distribution characteristics

### Benchmarking
- **Information Ratio**: Active return ÷ Tracking error
- **Beta**: Portfolio sensitivity to market
- **Tracking Error**: Standard deviation of active returns

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```
4. **Make your changes** with proper testing
5. **Commit and push**
   ```bash
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request**

### Code Standards

- **Type Hints**: Use modern Python typing
- **Documentation**: Add docstrings to all public functions
- **Testing**: Write tests for new features
- **Style**: Follow PEP 8 with Black formatting

### Adding New Strategies

```python
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult

class MyCustomStrategy(BaseStrategy):
    def generate_weights(self, data, **kwargs) -> StrategyResult:
        # Your strategy logic here
        pass
```
## 📝 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimer

**⚠️ NOT FINANCIAL ADVICE**

This software is for **educational and research purposes only**.

**Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.**

## 📚 References & Acknowledgments

### Academic Papers
- [Machine Learning for Stock Recommendation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3302088) - Machine learning approaches to stock selection
- [FinRL: Deep Reinforcement Learning Framework](https://arxiv.org/abs/2011.09607) - Deep RL framework for quantitative trading
- [Portfolio Allocation with Deep Reinforcement Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) - Portfolio optimization research

### Open Source Projects
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) - Deep reinforcement learning framework for quantitative trading
- [Alpaca-py](https://github.com/alpacahq/alpaca-py) - Alpaca trading API
- [bt](https://github.com/pmorissette/bt) - Flexible backtesting framework for Python

### Data Sources
- [Yahoo Finance](https://finance.yahoo.com/) - Free financial data
- [Financial Modeling Prep](https://financialmodelingprep.com/) - Professional financial data API
- [WRDS (Wharton Research Data Services)](https://wrds.wharton.upenn.edu/) - Academic financial database
- [Alpaca Markets](https://alpaca.markets/) - Brokerage API and market data

**Built with ❤️ for the quantitative finance community**

---
<div align="center">
<img align="center" width="30%" alt="image" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/e0371951-1ce1-488e-aa25-0992dafcc139">
</div>
