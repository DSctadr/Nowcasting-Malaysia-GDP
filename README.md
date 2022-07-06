# Nowcasting Malaysia’s GDP with Machine Learning

### Introduction
This report presents the International Data Science Accelerator Programme's output for Nowcasting Malaysia's Gross Domestic Product (GDP) using Machine Learning. This mentor-mentee programme was held over the 12 weeks (April 11 to July 1 2022) between mentees, Veronica Jamilat, Khadijah Jasni, Fatin Ezzati (statisticians from the Department of Statistics Malaysia) and Will Malpass (Data Scientist, Data Science Campus UK) as a mentor.

This project aims to identify a new and potential approach to nowcasting GDP using Machine Learning models. This new approach may complement the existing method of producing GDP advanced estimates. 

### Methodology Framework
![overall data science drawio](https://user-images.githubusercontent.com/58675575/175886895-ed878e5f-5225-4f8d-a26a-b9b5f25296f5.png)

### Data and Experimental Setup
The most crucial part of executing this project is acquiring relevant data series. Initially, we managed to compile a dataset comprising more than 100 economic-related variables at different time-frame. Some variables have more extended back series, while some indicators are only available in shorter time series, making the variable selection process consume much more time than expected. The process includes cross-checking available time-series data on different platforms and harmonising those datasets.
Fatin ezzati love choco
### Experimental Procedure
![Data Science drawio](https://user-images.githubusercontent.com/58675575/175885248-d361e44d-72b3-40f6-9f15-90f4b02e9d8b.png)


### Data
Before implementing Machine Learning Models, we have experimented with various combinations of a dataset. Those datasets are selected with variables that have extended back series and transformed into a stationary dataset. 

| Variable | Description | Units |
| ------------- | ------------- | ------------- |
| CPI  | Consumer Price Index | Index |
| CIC  | Coincident Index (CI) Composite Index  | Index |
|WSTIVO | WST Index (Volume) | Index |
|IPIELEC | IPI: Electricity | Index |
|WRTSICE	|Retail Sale of Information and Communication Equipment in Specialised Stores| RM Million |
|WRTS	|Wholesale and retail trade sales | RM Million |
|EXSITC0|	Exports: Food| RM Million |
|WRTSGRG	|Retail Sale of Cultural and Recreation Goods in Specialised Stores| RM Million |
|LOANBS	|Total Loans in Banking System| RM Million |
|MDTS|	Monthly Distributive Trade| RM Million |
|WSTSHHG|	Wholesale of Household Goods| RM Million |
|FIIXE|	Fixed Deposits| RM Million |
|WRTSAF| Retail Sale of Automotive Fuel in Specialised Stores| RM Million |
|WRTSOG| Retail Sale of Other Goods in Specialied Stores|  RM Million |
|LOANCB|	Total Loan Banking System (Fixed and Savings Deposits)| RM Million |
|IMSITC8|	Imports: Miscellaneous Manufactured Articles| RM Million |
|IPIMFG|	IPI: Manufacturing| Index |
|LGC|	Lagging Index (LG) Composite Index| Index |
|WRTSOHE|	475: Retail Sale of Other Household Equipment in Specialised Stores| RM Million |
|IPITOT|	IPI: Total| Index |
|DEPBANKSYS|	Total Deposits Banking System (Fixed and Savings Deposits)| RM Million | 
|FIXEIB|	Total Fixed Deposits Islamic Bank| RM Million |
|MSM2|	Money Supply: M2| RM Million |
|MSM3|	Money Supply: M3| RM Million |
|IMSITC5|	Imports: Chemicals| RM Million |
|LOANIB|	Total Loans Islamic Bank| RM Million |
|QCS|	Quarterly Construction Survey| RM Million |
|WRTSNSS|	Retail Sale in Non-Specialised Stores| RM Million |
|IMSITC0|	Imports: Food| RM Million |
|DEPOIB|	Total Deposits Islamic Bank| RM Million |
|IMTOT|	Imports: Total| RM Million |
|EXSITC5|	Exports: Chemicals| RM Million |
|WSTSFNB|	Wholesale of Food, Beverages and Tobacco| RM Million |
|WRTSFNB|	Retail Sale of Food, Beverages and Tobacco in Specialised Stores| RM Million |
|MSM1|	Money Supply: M1| RM Million | Index |
|LIC|	Leading Index (LI) Composite Index| Index |
|WSTSFEE|	Wholesale on a Fee or Contract Basis| RM Million |
|FIXECB|	Total Fixed Deposits Comm Banks| RM Million |
|EXTOT|	Exports: Total| RM Million |
|IMSITC6|	Imports: Manufactured Goods| RM Million |
|SWIPRO|	Swine Production| Number |
|EXSITC6|	Exports: Manufactured Goods| RM Million |
|EXSITC8|	Exports: Miscellaneous Manufactured Articles| RM Million |
|WSTSMES|	Wholesale of Machinery, Equipment and Supplies| RM Million |
|RIVA|	Retail Index (Value)| Index |
|RIVO|	Retail Index (Volume)| Index |
|AFRIB|	Average Financing Rate Islamic Bank| Percentage |
|DEPOIB2|	Total Deposits Islamic Bank| RM Million |
|IMSITC7|	Imports: Machinery & Transport Equipment| RM Million |
|IMSITC2|	Imports: Crude Materials, Inedible| RM Million |
|SLMFG|	Sales: Manufacturing sector| RM Million |
|SLENE|	Sales: Electrical and Electronic| RM Million |
|SAVDEP|	Savings Deposits| RM Million |
|IMSITC3|	Imports: Mineral Fuels, Lubricants, Etc.| RM Million |
|SLTEX|	Sales: Textiles, wearing apparel, leather products and footwear| RM Million |
|EXSITC7|	Exports: Machinery & Transport Equipment| RM Million |
|FOSWPRO|	Forestry: Sarawak| Cubic Metre |
|SLWOOD|	Sales: Wood products, furniture, paper products, printing and publishing| RM Million |
|EXSITC2|	Exports: Crude Materials, Inedible| RM Million |
|ALRCB|	Average Lending Rate Commercial Banks| Percentage | 
|FOSBPRO|	Forestry: Sabah| Cubic Metre |
|FOLOPRO|	Forestry: Malaysia| Cubic Metre |
|WSTSAGRI|	Wholesale of Agricultural Raw Materials and Livestock| RM Million |
|SDRIB|	Saving Deposits Interest Rate Islamic Bank| Percentage | 
|SLTRA|	Sales: Transport Equipment and Other Manufactures| RM Million |
|LOANMB|	Total Loans Merchant Banks| RM Million |
|SLNMET|	Sales: Non-metallic mineral products, basic metal and fabricated metal products| RM Million |
|IMSITC1|	Imports: Beverages And Tobacco| RM Million |
|CEMENT|	Cement Price| RM per 50 kg bag |
|FIAQUA|	Fishery: Aquaculture Production| Tonne Metric |
|WMVSMRM|	Maintenance and repair of motor vehicles| RM Million |
|USDEXC|	USD Exchange rate| USD |
|SLPET|	Sales: Petroleum, chemical, rubber and plastic products| RM Million | 
|EXSITC3|	Exports: Mineral Fuels, Lubricants, Etc.| RM Million | 
|WMVSMVA|	Sale of motor vehicle parts and accessories| RM Million | 
|PPID|	PPI: Electricity and gas| Index |
|EXSITC1|	Exports: Beverages And Tobacco| RM Million |
|JOBCREATE|	Job Openings| Thousands |
|WMVSMR|	Sale, maintenance and repair of motorcycles and related parts and accessories| RM Million |
|UERATE|	Unemployment Rate|  Percentage |
|PPITOT|	PPI: Total| Index |
|MVIVA|	Motor Vehicle Index (Value)| Index |
|IMSITC4|	Imports: Animal And Vegetable Oils And Fats| RM Million |
|PPIB|	PPI: Mining| Index |
|EXSITC4|	Exports: Animal And Vegetable Oils And Fats| RM Million |
|CRUDEBRUSD|	Crude oil (Brent)| USD per Barrel|
|IMSITC9|	Imports: Miscellaneous Transactions And Commodities| RM Million |
|PPIC|	PPI: Manufacturing| Index |
|IOSADMIN|	IoS-Administrative & Support Services| Index |
|IOSOTHER|	IoS-Other Private Services| Index |
|BFBLRIB|	Based Financing Rate (BLR) Islamic Banks| Percentage |
|BLBLRCB|	Based Lending Rate (BLR) Comm Banks| Percentage |
|PPIE|	PPI: Water supply| Index |
|EXSITC9|	Exports: Miscellaneous Transactions And Commodities| RM Million |
|PPIA|	PPI: Agriculture, forestry and fishing| Index |
|TOURIST|	Tourist Arrivals|  Number per thousand |
|IPIMIN|	IPI: Mining| Index |
|IOSTOT|	Index of Services: Total| Index |
|GDPGR|	GDP Growth Rate| RM Million |

