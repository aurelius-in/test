## ML Model Comparison Checklist

### Data Preparation
- [ ] Prepare Data: Preprocess the dataset (handle missing values, encode categorical variables, normalize features).
- [ ] Split Data: Divide the dataset into training and testing sets.

### Data Weighting
- [ ] Create two versions of the dataset: one with original weighting and another with adjusted weights (for class balance).

### Model Training and Evaluation
- [ ] Train and evaluate CatBoost model.
- [ ] Train and evaluate XGBoost model.
- [ ] Train and evaluate LightGBM model.
- [ ] Train and evaluate Random Forest model.
- [ ] Train and evaluate Logistic Regression model.
- [ ] Train and evaluate Support Vector Machine (SVM) model.
- [ ] Train and evaluate Gradient Boosting Machines (GBM) model.
- [ ] Train and evaluate AdaBoost model.
- [ ] Train and evaluate Decision Trees model.
- [ ] Train and evaluate Ridge Regression model.
- [ ] Train and evaluate Lasso Regression model.
- [ ] Train and evaluate TabNet model.

### Comparison and Analysis
- [ ] Compare each model's accuracy against CatBoost for weighted data.
- [ ] Compare each model's accuracy against CatBoost for unweighted data.
- [ ] Analyze which models outperform CatBoost under each condition.

### Visualization
- [ ] Plot a graph displaying the accuracy of each model in comparison to CatBoost for both weighted and unweighted data.

### Documentation
- [ ] Document findings and observations.

### Conclusions
- [ ] Draw conclusions based on the model comparisons.
- [ ] Make recommendations on the best models for the specific dataset and task.

