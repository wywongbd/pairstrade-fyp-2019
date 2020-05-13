## pairstrade-fyp-2019
Final year project at HKUST. We tested 3 main approaches for performing Pairs Trading: 
- distance method
- cointegration method (rolling OLS, Kalman Filter)
- reinforcement learning agent (proposed)

[Final report](https://github.com/wywongbd/statistical-arbitrage-18-19/blob/master/reports/FYP_Final_Report_LZ2.pdf).

FYP members: [myself](https://github.com/wywongbd), [Gordon](https://github.com/GordonCW), [Brendan](https://github.com/thambrendan)

### How to get started?
- Run `./setup.sh` to install all dependencies

### Note
- In our experiments, we used financial data taken from the Interactive Brokers platform, which is not free. Due to their regulations, we cannot release the financial data used in our experiments to the public. Feel free to use your own price data to perform experiments. 

### Disclaimer
- The strategies we implemented have not been proven to be profitable in a live trading account
- The reported returns are purely from backtesting procedures, and they may be susceptible to lookahead bias that we are not aware of

### Updates
- We're no longer developing this, check out [Yuri](https://github.com/ScrapeWithYuri/pairstrade-fyp-2019)'s findings regarding the RL agent
