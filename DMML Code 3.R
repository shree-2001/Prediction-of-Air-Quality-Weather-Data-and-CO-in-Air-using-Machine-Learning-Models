#Importing required libraries
library(tidyverse)
library(caret)
library(corrplot)
library(randomForest)
library(rpart)
library(ggplot2)
library(dplyr)
library(reshape2)
library(gbm)
library(e1071)
library(glmnet)
library(readxl)
# Step 1: Data Loading
# Importing the dataset
data <- read_excel("C:\\Users\\BhavS\\OneDrive\\Desktop\\Sem 1- project folder\\DMML-1\\Jayashree DMML Dataset\\owid-co2-data.xlsx")
#writing to CSV
write.csv(data, file = "C:\\Users\\BhavS\\OneDrive\\Desktop\\Sem 1- project folder\\DMML-1\\Jayashree DMML Dataset\\owid-co2-data.csv", row.names = FALSE)
#reading CSV file
df <- read.csv("C:\\Users\\BhavS\\OneDrive\\Desktop\\Sem 1- project folder\\DMML-1\\Jayashree DMML Dataset\\owid-co2-data.csv")
summary(df)
str(df)
#Step 2 : Data Cleaning
#Data-Type Converstion
df$iso_code <- as.numeric(factor(df$iso_code))
df$country <- as.numeric(factor(df$country))
df$year <- as.numeric(factor(df$year))
df$other_industry_co2 <- as.numeric(factor(df$other_industry_co2))
df$other_co2_per_capita <- as.numeric(factor(df$other_co2_per_capita))
df$share_global_other_co2 <- as.numeric(factor(df$share_global_other_co2))
df$cumulative_other_co2<- as.numeric(factor(df$cumulative_other_co2))
df$share_global_cumulative_other_co2 <- as.numeric(factor(df$share_global_cumulative_other_co2))
str(df)
#Step 3 : Data Exploration
#Visualization of the distribution of numeric variables
#Histogram
hist(df$co2)
#scatter Plot
plot(df$year, df$co2)
#Correlation Anlaysis
correlation_matrix <- cor(df[, c("co2", "year", "population", "gdp")])
print(correlation_matrix)
#heatmap
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

#Step 4: Data Preprocessing
# Imputation of missing values (if any)
# For simplicity, we'll use mean imputation for numeric variables
data <- df %>% mutate_if(is.numeric, ~if_else(is.na(.), mean(., na.rm = TRUE), .))

# Imputation of missing values (if any) for other numeric variables
for (col in names(df)[sapply(df, is.numeric)]) {
  if (anyNA(df[[col]])) {
    df[[col]][is.na(df[[col]])] <- mean(df[[col]], na.rm = TRUE)
    print(paste("Missing values imputed for column:", col))
  }
}
# Check if there are any missing values left
if (anyNA(data)) {
  print("There are still missing values in the dataset.")
} else {
  print("Data preprocessing completed successfully.")
}
# Remove rows with any missing values
preprocessed_data <- na.omit(df)
# Scale numeric variables
scaled_data <- scale(preprocessed_data[, c("co2", "year", "population", "gdp")])

#Step 4: Model fitting

# Define predictors (independent variables)
predictors <- df[, c("iso_code", "year", "population", "gdp")]

# Define target variable (dependent variable)
target <- df$co2

set.seed(123)
# Split data into training and testing sets

train_indices <- createDataPartition(data$co2, p = 0.8, list = FALSE)
train_data <- predictors[train_indices, ]
test_data <- predictors[-train_indices, ]
train_target <- target[train_indices]
test_target <- target[-train_indices]

#Model Fitting

#Model Fitting
# 1. Random Forest Regression model

rf_model <- randomForest(train_target ~ ., data = train_data, ntree = 100, mtry = sqrt(ncol(train_data)))

# 2. Gradient Boosting Regression

gbm_model <- gbm(train_target ~ ., data = train_data, distribution = "gaussian", n.trees = 100, interaction.depth = 3)

# 3. Support Vector Regression (SVR)

svr_model <- svm(train_target ~ ., data = train_data, kernel = "radial")

# 4. Ridge Regression Model

ridge_model <- glmnet(x = as.matrix(train_data), y = train_target, alpha = 0, lambda = 0.1)

# Print the summary of each model

summary(rf_model)
summary(gbm_model)
summary(svr_model)
summary(ridge_model)

# Evaluate models on the test set
# Make predictions on the testing data for each model

rf_predictions <- predict(rf_model, newdata = test_data)

gbm_predictions <- predict.gbm(gbm_model, newdata = test_data, n.trees = 100, type = "response")

svr_predictions <- predict(svr_model, newdata = test_data)

ridge_predictions <- predict(ridge_model, newx = as.matrix(test_data))

# Calculate Mean Squared Error (MSE) for each model

rf_mse <- mean((test_target - rf_predictions)^2)

gbm_mse <- mean((test_target - gbm_predictions)^2)

svr_mse <- mean((test_target - svr_predictions)^2)

ridge_mse <- mean((ridge_predictions - test_target)^2)

# Calculate Root Mean Squared Error (RMSE) for each model

rf_rmse <- sqrt(mean((test_target - rf_predictions)^2))

gbm_rmse <- sqrt(mean((test_target - gbm_predictions)^2))

svr_rmse <- sqrt(mean((test_target - svr_predictions)^2))

ridge_rmse <- sqrt(mean((ridge_predictions - test_target)^2))

# Calculate R-squared value for each model

rf_r_squared <- cor(test_target, rf_predictions)^2

gbm_r_squared <- cor(test_target, gbm_predictions)^2

svr_r_squared <- cor(test_target, svr_predictions)^2

ridge_residuals <- test_target - ridge_predictions
ridge_ssr <- sum(ridge_residuals^2)  # Sum of squared residuals
ridge_sst <- sum((test_target - mean(test_target))^2)  # Total sum of squares
ridge_r_squared <- 1 - (ridge_ssr / ridge_sst)  # Calculate R-squared

# Print evaluation metrics for Random Forest Regression Model

print("Evaluation Metrics for Random Forest Regression Model:")
print(paste("Mean Squared Error (MSE):", rf_mse))
print(paste("Root Mean Squared Error (RMSE):", rf_rmse))
print(paste("R-squared(rf):", rf_r_squared))
#Print evaluation metrics for Gradient Boosting Regression Model

print("Evaluation Metrics for Gradient Boosting Regression Model:")
print(paste("Mean Squared Error (MSE):", gbm_mse))
print(paste("Root Mean Square Error (RMSE):", gbm_rmse))
print(paste("R-squared(GBM)", gbm_r_squared))

#Print evaluation metrics for Support Vector regression Model

print("Evaluation Metrics for Support Vector Regression Model:")
print(paste("Mean Squared Error (MSE):", svr_mse))
print(paste("Root Mean Square Error (RMSE):", svr_rmse))
print(paste("R-Squared(SVR):", svr_r_squared))

#Print evaluation metrics for Ridge Regression Model

print("Evaluation Metrics for Neural Network Regression Model:")
print(paste("Mean Squared Error (MSE):", ridge_mse))
print(paste("Root Mean Square Error (RMSE):", ridge_rmse))
print(paste("R-Squraed(NNET):", ridge_r_squared ))

