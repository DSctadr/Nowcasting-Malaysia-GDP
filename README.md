# Nowcasting Malaysia’s GDP with Machine Learning

### Introduction
This report presents the International Data Science Accelerator Programme's output for Nowcasting Malaysia's Gross Domestic Product (GDP) using Machine Learning. This mentor-mentee programme was held over the 12 weeks (April 11 to July 1 2022) between mentees, Veronica Jamilat, Khadijah Jasni, Fatin Ezzati (statisticians from the Department of Statistics Malaysia) and Will Malpass (Data Scientist, Data Science Campus UK) as a mentor.

This project aims to identify a new and potential approach to nowcasting GDP using Machine Learning models. This new approach may complement the existing method of producing GDP advanced estimates. 

### Methodology Framework
![overall data science drawio](https://user-images.githubusercontent.com/58675575/175886895-ed878e5f-5225-4f8d-a26a-b9b5f25296f5.png)

### Data and Experimental Setup
The most crucial part of executing this project is acquiring relevant data series. Initially, we managed to compile a dataset comprising more than 100 economic-related variables at different time-frame. Some variables have more extended back series, while some indicators are only available in shorter time series, making the variable selection process consume much more time than expected. The process includes cross-checking available time-series data on different platforms and harmonising those datasets.

### Experimental Procedure
![Data Science drawio](https://user-images.githubusercontent.com/58675575/175885248-d361e44d-72b3-40f6-9f15-90f4b02e9d8b.png)


### Data
Before implementing Machine Learning Models, we have experimented with various combinations of a dataset. Those datasets are selected with variables that have extended back series and transformed into a stationary dataset. 

| Variable | Description |
| ------------- | ------------- |
| CPI  | Consumer Price Index |
| CIC  | Coincident Index (CI) Composite Index  |
|WSTIVO | WST Index (Volume) |
|IPIELEC | IPI: Electricity |
|WRTSICE	|Retail Sale of Information and Communication Equipment in Specialised Stores|
|WRTS	|Wholesale and retail trade sales |
|EXSITC0|	Exports: Food|
|WRTSGRG	|Retail Sale of Cultural and Recreation Goods in Specialised Stores|
|LOANBS	|Total Loans in Banking System|
|MDTS|	Monthly Distributive Trade|
|WSTSHHG|	Wholesale of Household Goods|
|FIIXE|	Fixed Deposits
|WRTSAF| Retail Sale of Automotive Fuel in Specialised Stores
|WRTSOG| Retail Sale of Other Goods in Specialied Stores
|LOANCB|	Total Loan Banking System (Fixed and Savings Deposits) 
|IMSITC8|	Imports: Miscellaneous Manufactured Articles
|IPIMFG|	IPI: Manufacturing
|LGC|	Lagging Index (LG) Composite Index
|WRTSOHE|	475: Retail Sale of Other Household Equipment in Specialised Stores
|IPITOT|	IPI: Total
|DEPBANKSYS|	Total Deposits Banking System (Fixed and Savings Deposits)
|FIXEIB|	Total Fixed Deposits Islamic Bank
|MSM2|	Money Supply: M2
|MSM3|	Money Supply: M3
|IMSITC5|	Imports: Chemicals
|LOANIB|	Total Loans Islamic Bank 
|QCS|	Quarterly Construction Survey
|WRTSNSS|	Retail Sale in Non-Specialised Stores
|IMSITC0|	Imports: Food
|DEPOIB|	Total Deposits Islamic Bank
|IMTOT|	Imports: Total
|EXSITC5|	Exports: Chemicals
|WSTSFNB|	Wholesale of Food, Beverages and Tobacco
|WRTSFNB|	Retail Sale of Food, Beverages and Tobacco in Specialised Stores
|MSM1|	Money Supply: M1
|LIC|	Leading Index (LI) Composite Index
|WSTSFEE|	Wholesale on a Fee or Contract Basis
|FIXECB|	Total Fixed Deposits Comm Banks
|EXTOT|	Exports: Total
|IMSITC6|	Imports: Manufactured Goods
|SWIPRO|	Swine Production
|EXSITC6|	Exports: Manufactured Goods
|EXSITC8|	Exports: Miscellaneous Manufactured Articles
|WSTSMES|	Wholesale of Machinery, Equipment and Supplies
|RIVA|	Retail Index (Value)
|RIVO|	Retail Index (Volume)
|AFRIB|	Average Financing Rate Islamic Bank
|DEPOIB2|	Total Deposits Islamic Bank
|IMSITC7|	Imports: Machinery & Transport Equipment
|IMSITC2|	Imports: Crude Materials, Inedible
|SLMFG|	Sales: Manufacturing sector
|SLENE|	Sales: Electrical and Electronic
|SAVDEP|	Savings Deposits
|IMSITC3|	Imports: Mineral Fuels, Lubricants, Etc.
|SLTEX|	Sales: Textiles, wearing apparel, leather products and footwear
|EXSITC7|	Exports: Machinery & Transport Equipment
|FOSWPRO|	Forestry: Sarawak
|SLWOOD|	Sales: Wood products, furniture, paper products, printing and publishing
|EXSITC2|	Exports: Crude Materials, Inedible
|ALRCB|	Average Lending Rate Commercial Banks 
|FOSBPRO|	Forestry: Sabah
|FOLOPRO|	Forestry: Malaysia
|WSTSAGRI|	Wholesale of Agricultural Raw Materials and Livestock
|SDRIB|	Saving Deposits Interest Rate Islamic Bank
|SLTRA|	Sales: Transport Equipment and Other Manufactures
|LOANMB|	Total Loans Merchant Banks
|SLNMET|	Sales: Non-metallic mineral products, basic metal and fabricated metal products
|IMSITC1|	Imports: Beverages And Tobacco
|CEMENT|	Cement Price 
|FIAQUA|	Fishery: Aquaculture Production
|WMVSMRM|	Maintenance and repair of motor vehicles
|USDEXC|	USD Exchange rate
|SLPET|	Sales: Petroleum, chemical, rubber and plastic products
|EXSITC3|	Exports: Mineral Fuels, Lubricants, Etc.
|WMVSMVA|	Sale of motor vehicle parts and accessories
|PPID|	PPI: Electricity and gas
|EXSITC1|	Exports: Beverages And Tobacco
|JOBCREATE|	Job Openings
|WMVSMR|	Sale, maintenance and repair of motorcycles and related parts and accessories
|UERATE|	Unemployment Rate
|PPITOT|	PPI: Total
|MVIVA|	Motor Vehicle Index (Value)
|IMSITC4|	Imports: Animal And Vegetable Oils And Fats
|PPIB|	PPI: Mining 
|EXSITC4|	Exports: Animal And Vegetable Oils And Fats
|CRUDEBRUSD|	Crude oil (Brent)
|IMSITC9|	Imports: Miscellaneous Transactions And Commodities
|PPIC|	PPI: Manufacturing
|IOSADMIN|	IoS-Administrative & Support Services
|IOSOTHER|	IoS-Other Private Services
|BFBLRIB|	Based Financing Rate (BLR) Islamic Banks
|BLBLRCB|	Based Lending Rate (BLR) Comm Banks
|PPIE|	PPI: Water supply
|EXSITC9|	Exports: Miscellaneous Transactions And Commodities
|PPIA|	PPI: Agriculture, forestry and fishing
|TOURIST|	Tourist Arrivals
|IPIMIN|	IPI: Mining
|IOSTOT|	Index of Services: Total
|GDPGR|	GDP Growth Rate

