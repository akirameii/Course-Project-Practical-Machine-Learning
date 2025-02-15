install.packages("caret")
install.packages("randomForest")
install.packages("e1071")  # Needed for confusion matrix
install.packages("ggplot2")

library(caret)
library(randomForest)
library(e1071)
library(ggplot2)

# Load the training and testing data
train_set <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
test_set <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))

# View the structure of the dataset
str(train_set)
str(test_set)

# Remove columns that contain mostly NA values
train_set <- train_set[, colSums(is.na(train_set)) == 0]
test_set <- test_set[, colSums(is.na(test_set)) == 0]

# Remove irrelevant columns (first 7 columns contain user ID and timestamps)
train_set <- train_set[, -c(1:7)]
test_set <- test_set[, -c(1:7)]

# Convert 'classe' into a factor variable
train_set$classe <- as.factor(train_set$classe)

# View dataset after cleaning
str(train_set)

set.seed(123)  # Ensure reproducibility
trainIndex <- createDataPartition(train_set$classe, p = 0.7, list = FALSE)

training_data <- train_set[trainIndex, ]
validation_data <- train_set[-trainIndex, ]

# Check distribution of classes
table(training_data$classe)
table(validation_data$classe)

set.seed(123)  # Ensure results are reproducible

# Train the model using Random Forest
model_rf <- randomForest(classe ~ ., data = training_data, importance = TRUE, ntree = 100)

# Print the model summary
print(model_rf)

# Ensure model is trained
set.seed(123)
model_rf <- randomForest(classe ~ ., data = training_data, importance = TRUE, ntree = 100)

# Make predictions on validation data
predictions_val <- predict(model_rf, newdata = validation_data)

# Generate confusion matrix
conf_matrix <- confusionMatrix(predictions_val, validation_data$classe)

# Print results
print(conf_matrix)

# Calculate accuracy
accuracy <- conf_matrix$overall["Accuracy"]
print(paste("Model Accuracy:", round(accuracy, 4)))

# Make predictions on test set
test_predictions <- predict(model_rf, newdata = test_set)

# Print test predictions
print(test_predictions)

# Save predictions
write.csv(test_predictions, "final_predictions.csv", row.names = FALSE)

# Save the trained model (optional)
saveRDS(model_rf, "final_model.rds")

