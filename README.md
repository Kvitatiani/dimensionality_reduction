# Dimensionality Reduction for Banking Campaign Analysis

## Project Overview

This project analyzes a **banking dataset** to identify the most important features in predicting whether a client will subscribe to a term deposit during a marketing campaign. Using **dimensionality reduction techniques**, the objective is to improve model performance and interpretability by focusing on the most relevant features.

## Objectives

- **Feature Selection & Reduction:** Identify key attributes while minimizing noise and redundancy.
- **Improve Predictive Accuracy:** Enhance the model's ability to predict client subscription outcomes.
- **Performance Optimization:** Reduce computational complexity for efficient modeling.

## About the Dataset

The dataset is based on the "Bank Marketing" dataset from the UCI Machine Learning Repository ([UCI Bank Marketing](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)).

Additionally, it includes five social and economic features sourced from Banco de Portugal ([Banco de Portugal Statistics](https://www.bportugal.pt/estatisticasweb)). This dataset closely resembles the one used in [Moro et al., 2014], though some attributes are omitted due to privacy concerns.

### Attribute Information

- **Bank client data:**
  1. age  
  2. job: type ("admin.","blue-collar", etc.)  
  3. marital: marital status ("divorced","married","single","unknown")  
  4. education: ("basic.4y","high.school","university.degree", etc.)  
  5. default: credit in default? ("no","yes","unknown")  
  6. housing: housing loan? ("no","yes","unknown")  
  7. loan: personal loan? ("no","yes","unknown")  

- **Last contact of current campaign:**
  8. contact: type ("cellular","telephone")  
  9. month: last contact month ("jan","feb", etc.)  
  10. day_of_week: last contact day ("mon","tue", etc.)  
  11. duration: last contact duration in seconds (important for benchmarking only)  

- **Campaign data:**
  12. campaign: contacts in this campaign  
  13. pdays: days since last contact from previous campaign (999 = not contacted)  
  14. previous: contacts before this campaign  
  15. poutcome: previous campaign outcome ("failure","nonexistent","success")  

- **Social/economic context:**
  16. emp.var.rate: employment variation rate  
  17. cons.price.idx: consumer price index  
  18. cons.conf.idx: consumer confidence index  
  19. euribor3m: euribor 3-month rate  
  20. nr.employed: number of employees  

- **Output variable:**
  21. y - has the client subscribed a term deposit? ("yes","no")  

## Techniques Used

The project applies the following dimensionality reduction techniques:

1. **Principal Component Analysis (PCA)**  
   - Reduces high-dimensional data while preserving variance.

2. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**  
   - Provides visualization of high-dimensional data by preserving local structures.

3. **Feature Importance Analysis**  
   - Identifies significant features using methods like Random Forest and Logistic Regression.

4. **Correlation Matrix & Variance Thresholding**  
   - Removes highly correlated and low-variance features to optimize model performance.

## Tools & Libraries Used

- **Python:** Core programming language.
- **Libraries:**
  - `pandas` – Data manipulation.
  - `numpy` – Numerical computations.
  - `scikit-learn` – Machine learning algorithms.
  - `matplotlib` / `seaborn` – Data visualization.
  - `plotly` – Interactive visualizations.

## Project Workflow

1. **Exploratory Data Analysis (EDA)**

2. **Data Preprocessing**

3. **Dimensionality Reduction**

4. **Conclusionn**



Follow the analysis in `dimensionality_reduction.ipynb`.


## Repository Structure

```

## Acknowledgments

Thanks to the UCI Machine Learning Repository and Banco de Portugal for providing the dataset.


