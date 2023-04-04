# Data Mining & Machine Learning Regression Model for Energy Intensity Reduction

**Team Members:**
- Laith Ajjan
- Connor Good
- Paul-Samuel Leitão
- Shruti Dua

**Advisor:** Alex Filstein P.Eng, ConocoPhillips

**Affiliation:** University of Calgary, Schulich School of Engineering

## Introduction

Steam assisted gravity drainage (SAGD) is widely used to extract oil from oil sands in Alberta. Non-Condensable Gas Co-Injection (NCG) uses Natural Gas Liquids (NGL) to reduce the amount of steam required for the process, reducing water and energy consumption. The project aims to analyze data from various SAGD sites and develop a machine learning regression model to forecast energy intensity reduction.

## Project Summary

The objective is to analyze monthly data from SAGD sites in Northern Alberta to determine the effect of NCG co-injection on oil extraction. The main parameter is the steam to oil ratio, with lower ratios indicating positive environmental impact. The model will help determine the most effective method for minimizing cost and environmental impact.

## Methods and Materials

### Software

AccuMap, a petroleum industry software, was used to extract monthly data from SAGD sites that use NCG and those that have not implemented NCG. 

### Machine Learning Model

Pandas and scikit-learn libraries were used to build the machine learning model. Well-site data extracted from AccuMap is transformed from .xls to .csv format for preprocessing in Python.

### Methodology

Data from facilities with NCG co-injection at SAGD sites are used to train the machine learning model. The model predicts values for similar sites, enabling cost-benefit analysis of NCG co-injection implementation. The model's quality is validated using the 𝑅^2 value and cross-validation.


### Results

![image](https://user-images.githubusercontent.com/77460425/229862607-28a1c42f-a3a5-4c98-b0d8-90349a862334.png)

By running the ModelUtilization.py script, a user can generate predicted steam injection values required for the site under optimal NCG conditions. This predicted steam injected required is then used in a plot with the actual values used to compare. The difference is then calculated to compute a percent steam injected saved for the site.

![image](https://user-images.githubusercontent.com/77460425/229849377-1482256d-6dc5-41df-b830-a3d232d5313c.png)

![image](https://user-images.githubusercontent.com/77460425/229849436-27b9aba2-3363-4a2e-9169-bd42ab664527.png)



## Abstract

Steam Assisted Gravity Drainage (SAGD) is a process used to extract heavy oil and bitumen from deep underground. Non-condensable gas (NCG) can reduce the amount of steam required, reducing the economic and environmental impacts. AccuMap was utilized to extract data from well sites in Northern Alberta. A machine learning regression model was created using Python to process the data and analyze the impacts of NCG implementation at existing SAGD sites.

