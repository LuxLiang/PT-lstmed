# PT-lstmed

Codes for Graduation Project in JLU, which is about 5min-frequency pair trading strategy. 

In this thesis, we explore two questions: (i)How to choose the right pair more effectively? (ii) How to choose the trading point more effectively? To solve the plm, we introduce OPTICS, ARMA, LSTM and Encoder-Decoder structure. 

For(i), we use density-based clustering to lower the time of calculation. What's more, Hurst exponent, half-period, cointegration, NZC(Number of Zero Crossing) are applied to find the right pair.

For(ii), we compare different models, which include GGR, ARMA, MLP, LSTM and LSTMED. LSTMED considered multi-step forecasting.
