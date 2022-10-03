# Nowcasting Malaysia’s GDP using Machine Learning

This project presents the International Data Science Accelerator Programme's output for Nowcasting Malaysia's Gross Domestic Product (GDP) using Machine Learning (ML). This mentor-mentee programme was held over 12 weeks (April 11 to July 1, 2022) between mentees, Veronica Jamilat, Khadijah Jasni, Fatin Ezzati (statisticians from the Department of Statistics Malaysia), and Will Malpass (Data Scientist, Data Science Campus UK) as a mentor.

### Introduction
As the COVID-19 pandemic unfolded, it became clear the need for a systematic analysis of high-frequency indicators of economic activity. The year 2020 saw the COVID-19 outbreak strike the world economy with unprecedented severity and unpredictability, providing a striking illustration of the need for timely knowledge (Barua, 2020). The concern emerged as to whether a wide range of higher frequency data from different sources could be properly analysed to deliver timely information on economic conditions that is useful for policymakers (Hopp, 2022). Therefore, the Department of Statistics Malaysia took the initiative to compile monthly advanced estimates to capture the extent of the contraction observed in the first half of 2020.  Thus, it is critical to give a current macroeconomic situation that goes beyond historical estimates and is capable of "nowcasting" current GDP in a timely and accurate manner (Proietti & Giovannelli, 2020). 

### Objective and Contribution
This project describes an effort to identify a new potential approach in nowcasting Malaysia’s GDP using ML and to complement (or replace) the existing method in producing Malaysia’s GDP advance estimates. It adds to the growing literature on nowcasting in several ways. First, it motivates and compiles datasets over 111 variables consisting of indexes, SITC-single digit code, banking, exchange rate and labour market. Second, we employ 11 machine learning (ML) algorithms to nowcast GDP growth. Third, the paper compares the performance of ML algorithms in dealing with and capturing extreme signal (s) by experimenting with different time frames (Full data: Q12005 - Q42021, COVID-19 exclude vaccination rollout: Q12020 – Q12021 and COVID-19 include vaccination rollout: Q12021 – Q42021). 

### Related Works
These are some of the countries that are working on nowcasting their countries GDP with ML. Based on table below, every country has different best models because each of the country has different economic conditions, different policy, people and culture and therefore, different data. Thus, not all models are applicable for each country.

| Countries | Selected Machine Learning Models | Best Model|
| ------------- | ------------- | ------------- |
|United Kingdom | LASSO, MIDAS, Random Forest, SVM, Neural Net, LSTM, DFM | LASSO |
|China  | SVM, DFM, Elastic Net, Random Forest and Gradient Boosting  | SVM |
|USA | LSTM, Bayesian VAR, Ridge, MIDAS, MLP, Random Forest, DFM, Gradient Boosting, Decision Tree, MF-VAR, OLS | LSTM, Bayesian VAR |
|New Zealand  | SVM, Neural Net, KNN, Boosted Trees, LASSO, Ridge, Elastic Net, DFM, BVAR, RBNZ | SVM & Neural Net |
|Austria | Ridge, Elastic Net, Lasso, Ridge, Random Forest, SVM, Neural Net, LSTM, BiLSTM, NNcon, Gaussian Process Regression | Ridge |
|Indonesia|	Random Forest, LASSO, Ridge, Elastic Net, Neural Networks, and Support Vector Machines|Random Forest |

### Methodology Framework
![overall data science-Page-1 drawio (3)](https://user-images.githubusercontent.com/104331591/193501760-5e353deb-018c-4d9e-a6d1-fae61912f8c1.png)

### Data and Experimental Setup
The most crucial part of executing this project is acquiring relevant data series. We initially managed to compile a dataset comprising more than 100 economic-related variables at different time frames. Some variables have more extended back series, while some indicators are only available in shorter time series, making the variable selection process consume much more time than expected. The process includes cross-checking available time-series data on different platforms and harmonising those datasets.

Currently, Malaysia's GDP advance estimate (AE) is compiled using the conventional method which requires many resources with more than 200 variables used. However, the data used are published with different frequencies whereas in this paper, we omitted variables that are quarterly data. The data consists of the time series of GDP and its 111 components taken from the production and expenditure quarterly and monthly economic indicators. This dataset consists of 106 monthly variables and 5 quarterly variables from the period 2005, January until 2021, December. This dataset involves three (3) sets of data sources: The Department of Statistics Malaysia (DOSM), the Central Bank of Malaysia (BNM) and the website (www.investing.com). 

### Dataset
Before implementing Machine Learning Models, we have experimented with various dataset combinations. Those datasets are selected with variables that have extended back series and transformed into a stationary dataset. List of 42 selected variables:

| Variable | Description | Units |
| ------------- | ------------- | ------------- |
|ALRCB | Average Lending Rate Commercial Banks | Rate |
|CIC  | Coincident Index (CI) Composite Index  | YoY (%) |
|CONSCRE | Consumption Credit | YoY (%) |
|CPI  | Consumer Price Index | Index |
|CRUDEBRUSD | Crude oil, Brent | YoY (%) |
|EXSITC0|	Exports: Food| RM Million |
|EXSITC1|	Exports: Beverages And Tobacco| RM Million |
|EXSITC2|	Exports: Crude Materials, Inedible| RM Million |
|EXSITC3|	Exports: Mineral Fuels, Lubricants, Etc.| RM Million | 
|EXSITC4|	Exports: Animal And Vegetable Oils And Fats| RM Million |
|EXSITC5|	Exports: Chemicals| RM Million |
|EXSITC6|	Exports: Manufactured Goods| RM Million |
|EXSITC7|	Exports: Machinery & Transport Equipment| RM Million |
|EXSITC8|	Exports: Miscellaneous Manufactured Articles| RM Million |
|EXSITC9|	Exports: Miscellaneous Transactions And Commodities| RM Million |
|EXTOT|	Exports: Total| RM Million |
|IMPCON |  Import of Consumption Goods | YoY (%) |
|IMSITC0|	Imports: Food| RM Million |
|IMSITC1|	Imports: Beverages And Tobacco| RM Million |
|IMSITC2|	Imports: Crude Materials, Inedible| RM Million |
|IMSITC3|	Imports: Mineral Fuels, Lubricants, Etc.| RM Million |
|IMSITC4|	Imports: Animal And Vegetable Oils And Fats| RM Million |
|IMSITC5|	Imports: Chemicals| RM Million |
|IMSITC6|	Imports: Manufactured Goods| RM Million |
|IMSITC7|	Imports: Machinery & Transport Equipment| RM Million |
|IMSITC8|	Imports: Miscellaneous Manufactured Articles| RM Million |
|IMSITC9|	Imports: Miscellaneous Transactions And Commodities| RM Million |
|IMTOT|	Imports: Total| RM Million | 
|IPIELEC | IPI: Electricity | Index |
|IPIMFG|	IPI: Manufacturing| Index |
|IPIMIN|	IPI: Mining| Index |
|IPITOT|	IPI: Total| Index |
|LGC|	Lagging Index (LG)| YoY (%) |
|LIC|	Leading Index (LI)| YoY (%) |
|OBOCC|	Outstanding Balance of Credit Cards| YoY (%) |
|PALMOIL|	Palm Oil| YoY (%) |
|RUBBER|	Rubber (SMR 20)| YoY (%) |
|SALETAX| Sales Tax| YoY (%) |
|SERVTAX| Service Tax| YoY (%) |
|SOPC| Sales of Passenger Cars| YoY (%) |
|TOURIST| Tourist Arrivals| YoY (%) |
|ACTIVEJOBS| Active Jobseekers| YoY (%) |
|JOBVACAN| Job Vacancy| YoY (%) |
|RETRENCH| Retrenchment| YoY (%) |
|USDEXC| USD Exchange rate| YoY (%) |

### Experiment flow
Processed data used in the modelling process includes 42 selected variables and transformed into a stationary dataset. There are three (3) datasets involved in the experiment. Dataset split into:

•	Full data (Q12005 - Q42021)

•	During COVID19, exclude  vaccination rollout (Q12020-Q12021)

•	During COVID19, include  vaccination rollout (Q12021-Q42021)

*Experimenting with different time-frame to see how those models perform with extreme signal(s) - side note: it is worth experimenting with those time frames because NY Fed shut down their nowcast over the pandemic!*

![image](https://user-images.githubusercontent.com/104331591/178622693-dd57d50c-43b9-46e8-a72d-7c3ba85ed5db.png)

### Results
Model performance – Comparison With and Without Rolling Window Method
This project use few dimensions to assess which ML models perform the best to nowcast Malaysia's GDP. We look at error models using RMSE and MAE and for stability and consistency using rolling window method and trendline to assess how close the actual GDP trendline with ML models'. 

![model performance (2)](https://user-images.githubusercontent.com/104331591/193489414-c42c19c6-5f77-4264-8e8c-ee429458ddd3.png)
As observed 


We conducted a rolling window method to cater small observation in the dataset and assess model stability. Rolling window relatively  improves model performance.
![image](https://user-images.githubusercontent.com/104331591/178670793-7259123c-47c7-4fa9-8044-a1d3b214930d.png)

Comparison of observed and predicted outcome
![image](https://user-images.githubusercontent.com/104331591/178882558-f19a4f3c-a703-47af-b7f4-b667445e59bc.png)
*Observed 1-year change in GDP and the predictions by the XG Boost, Random Forest, and the AR (left panel). Error of those methods (right panel)*

All three models underestimate GDP growth during the global financial crisis and pandemic. However, the XG Boost model is least biased (compared to other models) in those periods 

### Conclusion
The results of this project suggest that:
1)	ML models outperformed benchmark model (AR) in nowcasting Malaysia’s GDP
2)	XGBoost & Random Forest found to perform better than other ML models, both models perform better with shorter time-series (small observation) dataset and performed consistently, with and without extreme signal
3)	LightGBM performs better with extreme signal(s)

