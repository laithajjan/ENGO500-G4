# Data Mining & Machine Learning Regression Model for Energy Intensity Reduction

**Team Members:**
- Laith Ajjan
- Connor Good
- Paul-Samuel Leitão
- Shruti Dua

**Advisor:** Alex Filstein P.Eng, Conocophilips

**Affiliation:** University of Calgary, Schulich School of Engineering

## Introduction

Steam assisted gravity drainage (SAGD) is widely used to extract oil from oil sands in Alberta. Non-Condensable Gas Co-Injection (NCG) uses Natural Gas Liquids (NGL) to reduce the amount of steam required for the process, reducing water and energy consumption. The project aims to analyze data from various SAGD sites and develop a machine learning regression model to forecast energy intensity reduction.

## Project Summary

The objective is to analyze monthly data from SAGD sites in Northern Alberta to determine the effect of NCG co-injection on oil extraction. The main parameter is the steam to oil ratio, with lower ratios indicating positive environmental impact. The model will help determine the most effective method for minimizing cost and environmental impact.

## Methods and Materials

### Software

AccuMap, a petroleum industry software, was used to extract monthly data from SAGD sites that use NCG and those that have not implemented NCG. Parameters extracted include Cumulative Steam Injected, Cumulative Oil Produced, Number of well pairs, etc.

### Machine Learning Model

Pandas and scikit-learn libraries were used to build the machine learning model. Well-site data extracted from AccuMap is transformed from .xls to .csv format for preprocessing in Python.

### Methodology

Data from facilities with NCG co-injection at SAGD sites are used to train the machine learning model. The model predicts values for similar sites, enabling cost-benefit analysis of NCG co-injection implementation. The model's quality is validated using the 𝑅^2 value.

## Abstract

Steam Assisted Gravity Drainage (SAGD) is a process used to extract heavy oil and bitumen from deep underground. Non-condensable gas (NCG) can reduce the amount of steam required, reducing the economic and environmental impacts. AccuMap was utilized to extract data from well sites in Northern Alberta. A machine learning regression model was created using Python to process the data and analyze the impacts of NCG implementation at existing SAGD sites.

