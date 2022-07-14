# Nowcasting Malaysia’s GDP using Machine Learning

### Introduction
This report presents the International Data Science Accelerator Programme's output for Nowcasting Malaysia's Gross Domestic Product (GDP) using Machine Learning (ML). This mentor-mentee programme was held over 12 weeks (April 11 to July 1, 2022) between mentees, Veronica Jamilat, Khadijah Jasni, Fatin Ezzati (statisticians from the Department of Statistics Malaysia), and Will Malpass (Data Scientist, Data Science Campus UK) as a mentor.

### Motivation
Accurate and timely information is crucial in facilitating decision-making. As practised in many countries, advance estimate (AE) of GDP is generally available in the first month after each quarter. 
Currently, Malaysia's GDP AE is compiled using the conventional method, which requires many resources, and is only used internally. Recent literatures have proven ML to outperform Autoregressive (AR) as a benchmark model in nowcasting (forecasting) GDP (and other economic indicators)
Therefore, this project aims to identify a new and potential approach in nowcasting Malaysia's GDP using ML and to complement (or replace) the existing method in producing Malaysia's GDP advance estimates.

### Methodology Framework
![overall data science drawio](https://user-images.githubusercontent.com/58675575/175886895-ed878e5f-5225-4f8d-a26a-b9b5f25296f5.png)

### Data and Experimental Setup
The most crucial part of executing this project is acquiring relevant data series. We initially managed to compile a dataset comprising more than 100 economic-related variables at different time frames. Some variables have more extended back series, while some indicators are only available in shorter time series, making the variable selection process consume much more time than expected. The process includes cross-checking available time-series data on different platforms and harmonising those datasets.

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
|USDEXC| USD Exchange rate| YoY (%) |

### Experiment flow
Processed data used in the modelling process includes 42 selected variables and transformed into a stationary dataset. There are three (3) datasets involved in the experiment. Dataset split into:

•	Full data (Q12005 - Q42021)

•	During COVID19, exclude  vaccination rollout (Q12020-Q12021)

•	During COVID19, include  vaccination rollout (Q12021-Q42021)

*Experimenting with different time-frame to see how those models perform with extreme signal(s) - side note: it is worth experimenting with those time frames because NY Fed shut down their nowcast over the pandemic!*

![image](https://user-images.githubusercontent.com/104331591/178622693-dd57d50c-43b9-46e8-a72d-7c3ba85ed5db.png)

### Results

*Model performance – Full data without applying the rolling window method*
![image](https://user-images.githubusercontent.com/104331591/178670548-64a1ad88-4150-47ad-8286-c9b90c5ae03d.png)

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

