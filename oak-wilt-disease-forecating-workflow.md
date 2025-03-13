# Oak Wilt Disease Prediction: A Data-Driven Approach

## Overview
This research develops a predictive framework for early detection of Oak Wilt Disease (OWD) using remote sensing and climate data. By integrating Sentinel-2 imagery with daily climate variables, we identify key environmental conditions that trigger disease onset, allowing for proactive management interventions before visible symptoms appear.

## Workflow Diagram

```mermaid
flowchart TB
    subgraph Data["Data Acquisition & Processing"]
        A1["Remote Sensing Data
Sentinel-2 Imagery"] --> A2["Extract NDVI
Time Series"]
        A3["Climate Data
Temperature, Humidity, Wind,
Precipitation"] --> A4["Process Daily
Climate Variables"]
        A5["Oak Wilt
Sample Data"] --> A6["Prepare Reference
Dataset"]
    end
    
    subgraph Feature["Feature Engineering"]
        A2 --> B1["Estimate Start of Season
SOS Detection"]
        B1 --> B2["Calculate SOS-Relative
Temporal Windows"]
        B2 --> B3["Extract Climate Variables
for Each Window"]
        A4 --> B3
        B3 --> B4["Generate Additional Features
Consecutive Hot Days
Low Humidity Days"]
    end
    
    subgraph Select["Feature Selection & Analysis"]
        B4 --> C1["Ensemble Variable
Importance Analysis"]
        C1 --> C2["Select Top 50
Predictive Variables"]
        C2 --> C3["CART Regression Tree
Analysis"]
        C3 --> C4["Identify Key Thresholds
Tmax > 25°C, RH < 65%"]
    end
    
    subgraph Model["Predictive Modeling"]
        C4 --> D1["Split Training
& Testing Data"]
        D1 --> D2["Train Multiple Models"]
        D2 --> D3["XGBoost"]
        D2 --> D4["Random Forest"]
        D2 --> D5["SVM"]
        D2 --> D6["Neural Network"]
        D2 --> D7["Logistic Regression"]
        D3 & D4 & D5 & D6 & D7 --> D8["Evaluate Model
Performance"]
        D8 --> D9["Select Optimal
Prediction Model"]
    end
    
    subgraph Predict["Prediction & Validation"]
        D9 --> E1["Predict OWD
Onset Probability"]
        E1 --> E2["Generate Spatial
Prediction Maps"]
        E2 --> E3["Validate with
Ground Data"]
        E3 --> E4["Calculate Performance
Metrics"]
    end
    
    A6 --> D1
    E4 -.-> F1["Apply Model to
New Areas"]
    
    classDef default fill:transparent,stroke:#333,stroke-width:2px,color:#000,font-weight:bold,rx:10,ry:10
    classDef featureClass fill:transparent,stroke:#228B22,stroke-width:2px,color:#228B22,font-weight:bold,rx:10,ry:10
    classDef dataSource fill:transparent,stroke:#2f73df,stroke-width:2px,color:#2f73df,font-weight:bold,rx:10,ry:10
    classDef analysis fill:transparent,stroke:#f2a93b,stroke-width:2px,color:#f2a93b,font-weight:bold,rx:10,ry:10
    classDef model fill:transparent,stroke:#82b366,stroke-width:2px,color:#82b366,font-weight:bold,rx:10,ry:10
    classDef result fill:transparent,stroke:#9673a6,stroke-width:2px,color:#9673a6,font-weight:bold,rx:10,ry:10
    
    class A1,A3,A5 dataSource
    class B1,B2,B3,B4 featureClass
    class C1,C2,C3,C4 analysis
    class D1,D2,D3,D4,D5,D6,D7,D8,D9 model
    class E1,E2,E3,E4,F1 result
    
    Data:::dataSource
    Feature:::featureClass
    Select:::analysis
    Model:::model
    Predict:::result
```

## Methodology Details

### 1. Data Acquisition & Processing
- **Remote Sensing**: Sentinel-2 multispectral imagery (10-20m resolution) collected throughout the growing season
- **Climate Data**: Daily measurements of temperature (Tmax, Tmin), relative humidity, precipitation, wind speed and direction
- **Reference Data**: Field observations of OWD presence/absence across 320 plots, with onset dates recorded for positive cases

### 2. Feature Engineering
- **Start of Season (SOS) Detection**: Novel approach using local minima in NDVI time series to identify leaf emergence timing
- **Temporal Windows**: Define 10 temporal windows (5 before SOS, 5 after SOS) to capture relevant climate conditions
- **Climate Feature Extraction**: Aggregate climate variables over each temporal window (mean, max, min, standard deviation)
- **Specialized Features**: Create specific indicators for consecutive hot days, low humidity periods, and combined hot-dry conditions

### 3. Variable Selection & Threshold Identification
- **Ensemble Feature Importance**: Combine variable rankings from multiple models (XGBoost, Random Forest, Logistic Regression)
- **Dimensionality Reduction**: Select top 50 variables with highest predictive power
- **CART Analysis**: Identify critical thresholds and decision rules through decision tree modeling
- **Key Findings**: Maximum temperature >25°C for ≥3 consecutive days + relative humidity <65% within 10 days after SOS

### 4. Predictive Modeling
- **Model Evaluation**: Compare five machine learning approaches (XGBoost, Random Forest, SVM, Neural Network, Logistic Regression)
- **Dual Prediction Tasks**: 
  1. Classification: Will OWD occur? (yes/no)
  2. Regression: When will OWD onset occur? (day of year)
- **Performance Metrics**: Accuracy, F1-score, RMSE, and success ratio within ±8 days window
- **Best Model**: XGBoost (67.2% success rate, RMSE of 7 days for onset prediction)

### 5. Spatial Prediction & Validation
- **Prediction Maps**: Generate probability surfaces showing likelihood of OWD onset
- **Uncertainty Assessment**: Identify areas with less reliable predictions
- **Validation**: Compare predictions with field observations
- **Application**: Inform targeted management interventions in high-risk areas

## Key Findings

1. **Critical Climate Triggers**: The combination of high temperature (>25°C) and low humidity (<65%) for at least three consecutive days following leaf emergence creates optimal conditions for OWD onset.

2. **Temporal Sensitivity**: The 10-day window immediately after the Start of Season represents the most critical period for disease prediction.

3. **Model Performance**: The XGBoost algorithm demonstrated superior performance in predicting both OWD occurrence (83.1% accuracy) and onset timing (RMSE of 7 days).

4. **Early Warning Potential**: This approach enables prediction of OWD risk 10-30 days before visible symptoms appear, significantly expanding the window for preventive interventions.

## Significance

This research advances forest health monitoring by:
1. Developing a reproducible methodology for early disease detection
2. Identifying specific climate thresholds that trigger OWD onset
3. Creating a framework applicable across broad spatial scales
4. Enabling proactive rather than reactive management strategies

The integration of remote sensing with climate data offers a powerful approach for monitoring forest health in the face of emerging threats and changing climate conditions.
