# Importing required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(ggplot2)
library(reshape2)
library(dplyr)
library(gbm)
library(e1071)
library(nnet)

# Step 1: Data reading and analyzing
# Importing the dataset
df <- read.csv("C:\\Users\\BhavS\\OneDrive\\Desktop\\Sem 1- project folder\\DMML-1\\Jayashree DMML Dataset\\weatherHistory.csv")
summary(df)
str(df)

#Step 2: Converting Char values to numeric values
df$Formatted.Date <- as.numeric(factor(df$Formatted.Date))
df$Summary <- as.numeric(factor(df$Summary))
df$Precip.Type <- as.numeric(factor(df$Precip.Type))
df$Daily.Summary <- as.numeric(factor(df$Daily.Summary))
summary(df)
str(df)

#Step 3: Data Exploration
# Visualization of distributions
# Histogram of Temperature
ggplot(df, aes(x = Temperature..C.)) +
  geom_histogram(binwidth = 2, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Temperature", x = "Temperature (Celsius)", y = "Frequency")

# Boxplot of Humidity
ggplot(df, aes(y = Humidity)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Distribution of Humidity", x = "", y = "Humidity")

# Scatterplot of Temperature vs. Apparent Temperature
ggplot(df, aes(x = Temperature..C., y = Apparent.Temperature..C.)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Temperature vs. Apparent Temperature", x = "Temperature (Celsius)", y = "Apparent Temperature (Celsius)")


#Calculating  Correlation matrix
correlation_matrix <- cor(df[,c("Temperature..C.", "Apparent.Temperature..C.", "Humidity", "Wind.Speed..km.h.")])
print(correlation_matrix)

#Visualizing the correlation matrix as a heatmap

#Converting correlation matrix to long format for plotting
correlation_df<-melt(correlation_matrix)

#Plotting heatmap
heatmap <- ggplot(correlation_df, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high= "red", midpoint = 0, limits = c(-1,1),
                       name = "Correlation") + 
  theme_minimal()+
  labs(title = "Correlation Heatmap", x = "", y = "")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(heatmap)

#Save the plot as an image file (e.g., PNG format)
ggsave("Correlation Matrix.png", plot = heatmap, width = 10, height = 6, dpi = 300)

#Step 4: Data Preprocessing
# Calculate the interquartile range (IQR) for each numeric variable
Q1 <- apply(df[, c("Temperature..C.", "Apparent.Temperature..C.", "Humidity", "Wind.Speed..km.h.")], 2, quantile, probs = 0.25)
Q3 <- apply(df[, c("Temperature..C.", "Humidity", "Wind.Speed..km.h.", "Pressure..millibars.")], 2, quantile, probs = 0.75)
IQR <- Q3 - Q1
# Define the threshold for outlier detection (e.g., 1.5 times the IQR)
threshold <- 1.5
# Identify outliers
outliers <- apply(df[, c("Temperature..C.", "Humidity", "Wind.Speed..km.h.", "Pressure..millibars.")], 2,
                  function(x) {
                    lower_bound <- Q1 - threshold * IQR
                    upper_bound <- Q3 + threshold * IQR
                    x < lower_bound | x > upper_bound
                  })

# Imputation of missing values (if any)
# For simplicity, we'll use mean imputation for numeric variables
data <- df %>% mutate_if(is.numeric, ~if_else(is.na(.), mean(., na.rm = TRUE), .))

# Check if there are any missing values left
if (anyNA(data)) {
  print("There are still missing values in the dataset.")
} else {
  print("Data preprocessing completed successfully.")
}

#Step 5: Feature Selection/Engineering
# Feature Selection:
# Since the dataset is already condensed and contains relevant weather features,
# we can skip explicit feature selection for now.

# Feature Engineering:
# We can create new features based on domain knowledge or patterns observed in the data.

# Example: Creating a new feature for the difference between Temperature and Apparent Temperature
data <- df %>% mutate(Temperature_Difference = Temperature..C. - Apparent.Temperature..C.)

# Example: Creating a new feature for the product of Humidity and Wind Speed
data <- df %>% mutate(Humidity_Wind_Product = Humidity * Wind.Speed..km.h.)

# Example: Creating a new feature for the interaction between Temperature and Humidity
data <- df %>% mutate(Temperature_Humidity_Interaction = Temperature..C. * Humidity)

# Example: Creating a new feature for the ratio of Temperature to Pressure
data <- df %>% mutate(Temperature_Pressure_Ratio = Temperature..C. / Pressure..millibars.)

# Check the updated structure of the data
str(data)

#Step 6: Model Fitting
# Define predictors and target variable

predictors <- select(df, -Formatted.Date, -Daily.Summary)  # Exclude Formatted.Date and Daily.Summary
target <- df$Temperature..C.  # Assuming Temperature..C. is the target variable
# Split the data into training and testing sets

set.seed(123)  # For reproducibility
train_index <- createDataPartition(target, p = 0.8, list = FALSE)
train_data <- predictors[train_index, ]
test_data <- predictors[-train_index, ]
train_target <- target[train_index]
test_target <- target[-train_index]
#Model fitting

# Remove the "Loud.Cover" column from both the training and testing datasets
train_data <- subset(train_data, select = -Loud.Cover)
test_data <- subset(test_data, select = -Loud.Cover)


# 1. Random Forest Regression model

rf_model <- randomForest(train_target ~ ., data = train_data)


# 2. Gradient Boosting Regression

gbm_model <- gbm(train_target ~ ., data = train_data, distribution = "gaussian", n.trees = 100, interaction.depth = 3)

# 3. Support Vector Regression (SVR)

svr_model <- svm(train_target ~ ., data = train_data, kernel = "radial")

# 4. Neural Network Regression

nn_model <- nnet(train_target ~ ., data = train_data, size = 10, linout = TRUE)


# Print the summary of each model

summary(rf_model)
summary(gbm_model)
summary(svr_model)
summary(nn_model)

# Evaluate models on the test set
# Make predictions on the testing data for each model

rf_predictions <- predict(rf_model, newdata = test_data)

gbm_predictions <- predict.gbm(gbm_model, newdata = test_data, n.trees = 100, type = "response")

svr_predictions <- predict(svr_model, newdata = test_data)

nn_predictions <- predict(nn_model, newdata = test_data)

# Calculate Mean Squared Error (MSE) for each model

rf_mse <- mean((test_target - rf_predictions)^2)

gbm_mse <- mean((test_target - gbm_predictions)^2)

svr_mse <- mean((test_target - svr_predictions)^2)

nn_mse <- mean((test_target - nn_predictions)^2)

# Calculate Root Mean Squared Error (RMSE) for each model

rf_rmse <- sqrt(mean((test_target - rf_predictions)^2))

gbm_rmse <- sqrt(mean((test_target - gbm_predictions)^2))

svr_rmse <- sqrt(mean((test_target - svr_predictions)^2))

nn_rmse <- sqrt(mean((test_target - nn_predictions)^2))

# Calculate R-squared value for each model

rf_r_squared <- cor(test_target, rf_predictions)^2

gbm_r_squared <- cor(test_target, gbm_predictions)^2

svr_r_squared <- cor(test_target, svr_predictions)^2

nn_r_squared <- cor(test_target, nn_predictions)^2

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

#Print evaluation metrics for Neural Network Regression Model

print("Evaluation Metrics for Neural Network Regression Model:")
print(paste("Mean Squared Error (MSE):", nn_mse))
print(paste("Root Mean Square Error (RMSE):", nn_rmse))
print(paste("R-Squraed(NNET):", nn_r_squared ))









