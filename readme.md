# Data Mining & Machine Learning Regression Model for Energy Intensity Reduction

**Team Members:**
- Laith Ajjan
- Connor Good
- Paul-Samuel Leitão
- Shruti Dua

**Advisor:** Alex Filstein P.Eng, ConocoPhillips

**Affiliation:** University of Calgary, Schulich School of Engineering

## Introduction

Steam assisted gravity drainage (SAGD) is widely used to extract oil from oil sands in Alberta. Non-Condensable Gas Co-Injection (NCG) uses Natural Gas Liquids (NGL) to reduce the amount of steam required for the process, reducing water and energy consumption. This project aims to analyze data from various SAGD sites and develop a machine learning regression model to forecast energy intensity reduction.

## Project Summary

The objective is to analyze monthly data from SAGD sites in Northern Alberta to determine the effect of NCG co-injection on oil extraction. The main parameter is the steam to oil ratio, with lower ratios indicating positive environmental impact. The model will help determine the most effective method for minimizing cost and environmental impact.

## Methods and Materials

### Software

AccuMap, a petroleum industry software, was used to extract monthly data from SAGD sites that use NCG and those that have not implemented NCG. 

### Machine Learning Model

Pandas and scikit-learn libraries were used to build the machine learning model. Well-site data extracted from AccuMap is transformed from .xls to .csv format for preprocessing in Python.

### Methodology

Data from facilities with NCG co-injection at SAGD sites are used to train the machine learning model. The model predicts values for similar sites, enabling cost-benefit analysis of NCG co-injection implementation. The model's quality is validated using the RMSE value and cross-validation.

## Results

### Model Performance
![image](https://user-images.githubusercontent.com/52933277/236649952-4606d2f1-26d4-4d60-b2b4-f78b499e71ea.png)

The RandomForestRegressor model was used for predicting steam injection requirements. The model achieved a train score of 0.801 and a test score of 0.784. Cross-validation scores ranged from 0.630 to 0.760, indicating reasonable consistency across different subsets of the training data. The model's root mean squared error (RMSE) was 0.414, and the normalized root mean squared error (NRMSE) was 0.077.


### Feature Importance
![image](https://user-images.githubusercontent.com/52933277/236649959-a5a65948-c5d0-467b-851d-96f71c7133c7.png)

Feature importance analysis revealed that all features contributed to the model's predictive ability, with 'CalDlyOil(m3/d)' having the highest importance. The other features, such as 'NCG/steam', 'PrdHours(hr)', 'NbrofWells', and 'InjHours(hr)', also played significant roles in determining steam requirements.

### Partial Dependence of SOR on NCG/Steam
![image](https://user-images.githubusercontent.com/52933277/236649975-03c9db48-8147-4e1b-a188-4be5a0edd8c3.png)

The partial dependence plot showed a clear relationship between the steam-to-oil ratio and the NCG-to-steam ratio. As the NCG/steam ratio increased, the steam-to-oil ratio decreased, indicating the effectiveness of NCG co-injection in reducing steam consumption and its associated environmental and economic impacts.


For the SAGD site Bolney, which does not use NCG co-injection, we see that the model accurately predicts the total steam required during the life cycle of the site to about 2% cumulative steam injected over the sites. Then, using the optimal NCG co-injection scheme from the Wabiskaw site, the model predicts that 29% of the steam injected could have been saved. Keep in mind that part of the Bolney site was used to train the model, so the model has seen some of these data points during training.

![image](https://user-images.githubusercontent.com/52933277/236650085-32e790cd-5ac4-4cd7-a0ba-e8f962c3ad3d.png)
![image](https://user-images.githubusercontent.com/52933277/236650089-c173b8b8-9a74-4259-ae65-da3ddd3c6707.png)

### Steam Injection Savings

The model was applied to various SAGD sites, estimating the potential savings in steam injection with the implementation of NCG co-injection. Results showed significant potential savings, ranging from 29% to 38% for different sites. This demonstrates the potential benefits of NCG co-injection for reducing the environmental and economic impacts of SAGD operations.

![image](https://user-images.githubusercontent.com/52933277/236650119-2e4ba793-3ffc-4cd3-aee2-14aeda0eba73.png)
![image](https://user-images.githubusercontent.com/52933277/236650134-a1c26ef2-b093-4de1-aa65-74a237cbc75b.png)


## Conclusion

The developed machine learning regression model successfully analyzed the impact of NCG co-injection on steam-to-oil ratios in SAGD operations. The model showed the potential for significant savings in steam injection, leading to reduced energy consumption and environmental impact. This information can help oil companies make informed decisions about implementing NCG co-injection in their SAGD operations.
