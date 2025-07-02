#Importing required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(ggplot2)
library(dplyr)
library(reshape2)
library(gbm)
library(e1071)
library(glmnet)
# Step 1: Data Loading
# Importing the dataset
df <- read.csv("C:\\Users\\BhavS\\OneDrive\\Desktop\\Sem 1- project folder\\DMML-1\\Jayashree DMML Dataset\\Air_Quality.csv")
summary(df)
str(df)
# Step 2: Data Cleaning
#Converting Data Types
df$Unique.ID <- as.numeric(factor(df$Unique.ID))
df$Indicator.ID <- as.numeric(factor(df$Indicator.ID))
df$Name <- as.numeric(factor(df$Name))
df$Measure <- as.numeric(factor(df$Measure))
df$Measure.Info <- as.numeric(factor(df$Measure.Info))
df$Geo.Type.Name <- as.numeric(factor(df$Geo.Type.Name))
df$Geo.Join.ID <- as.numeric(factor(df$Geo.Join.ID))
df$Time.Period <- as.numeric(factor(df$Time.Period))
df$Geo.Place.Name <- as.numeric(factor(df$Geo.Place.Name))
df$Time.Period <- as.numeric(factor(df$Time.Period))
df$Start_Date <- as.numeric(factor(df$Start_Date))
df$Message <- as.numeric(factor(df$Message))

str(df)
#Setp 2: Data Exploration
#Histogram of Data.Value
hist(df$Data.Value, main = "Distribution of Data.Values", xlab = "Data.Value")

#Scatter plot of Data.Value vs. Start_Date
plot(df$Start_Date, df$Data.Value, main = "Data.Value vs. Start_Date", xlab = "Start_Date", ylab = "Data.Value")

#Time series plot of Data.Value
plot(df$Start_Date, df$Data.Value, type = "l", main = "Time Series Plot of Data.Value", xlab = "Start_Date", ylab = "Data.Value")

#Bar plot of Name
barplot(table(df$Name), main = "Frequency of Name", xlab = "Name")

#Correlation Analysis
#Importing required libraries
library(corrplot)
# Compute correlation matrix
correlation_matrix <- cor(df[, c("Unique.ID", "Indicator.ID", "Name", "Measure", "Measure.Info", 
                                 "Geo.Type.Name", "Geo.Join.ID", "Geo.Place.Name", 
                                 "Time.Period", "Start_Date", "Data.Value")])
print(correlation_matrix)

# Visualize correlation matrix as heatmap
heatmap(correlation_matrix, 
        Colv = NA, Rowv = NA,         # Turn off row and column clustering
        col = colorRampPalette(c("blue", "white", "red"))(100),  # Color palette
        scale = "none",               # No scaling
        margins = c(5, 5),            # Add margins
        main = "Correlation Heatmap", # Title
        xlab = "Variables", ylab = "Variables")  # Labels for x and y axis

# Add correlation values to the heatmap
for(i in 1:nrow(correlation_matrix)) {
  for(j in 1:ncol(correlation_matrix)) {
    text(j, i, round(correlation_matrix[i, j], 2), cex = 0.8, col = "black", font = 2)
  }
}

#Step 3: Data Preprocessing
# Check if "Data.Value" is a numeric column
if ("Data.Value" %in% names(df) && is.numeric(df$Data.Value)) {
  # Check for missing values
  if (anyNA(df$Data.Value)) {
    # Impute missing values with the mean
    mean_value <- mean(df$Data.Value, na.rm = TRUE)
    df$Data.Value[is.na(df$Data.Value)] <- mean_value
  }
  
# Calculate the interquartile range (IQR) for "Data.Value"
Q1 <- quantile(df$Data.Value, probs = 0.25, na.rm = TRUE)
Q3 <- quantile(df$Data.Value, probs = 0.75, na.rm = TRUE)
IQR <- Q3 - Q1
  
# Define the threshold for outlier detection (e.g., 1.5 times the IQR)
threshold <- 1.5
  
# Identify outliers
outliers <- df$Data.Value < (Q1 - threshold * IQR) | df$Data.Value > (Q3 + threshold * IQR)
  
# Imputation of missing values (if any) for other numeric variables
for (col in names(df)[sapply(df, is.numeric)]) {
  if (anyNA(df[[col]])) {
    df[[col]][is.na(df[[col]])] <- mean(df[[col]], na.rm = TRUE)
    print(paste("Missing values imputed for column:", col))
  }
}
  
# Check if there are any missing values left
if (anyNA(df)) {
  print("There are still missing values in the dataset.")
  print(colSums(is.na(df)))
} else {
  print("Data preprocessing completed successfully.")
}
} else {
  print("Data preprocessing failed. 'Data.Value' column is either missing or not numeric.")
}
#Seem like there are still missing values
#Removing the Column Message 
data <- subset(df, select = -Message)

#Again checking if there are any missing values left
if(anyNA(data)){
  print("There are still missing values in the dataset.")
  print(colSums(is.na(data)))
} else {
  print("Data preprocessing completed successfully.")
}
str(data)

#Step 4: Model Fitting

predictors <- c("Unique.ID", "Indicator.ID", "Name", "Measure", "Measure.Info", 
                "Geo.Type.Name", "Geo.Join.ID", "Geo.Place.Name", 
                "Time.Period", "Start_Date")

# Target variable (dependent variable)
target <- "Data.Value"
#Splitting the data into training and test set(80% for training and 20% for testing)
# Set seed for reproducibility
set.seed(123)
# Split data into training and testing sets

train_indices <- createDataPartition(data$Data.Value, p = 0.8, list = FALSE)
train_data <- data[train_indices, predictors]
test_data <- data[-train_indices, predictors]
train_target <- data[train_indices, target]
test_target <- data[-train_indices, target]

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





