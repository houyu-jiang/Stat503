library(MASS) 
library(rpart)
library(rattle)
library(tidyverse)
library(randomForest)
library(gbm)

# Data
train <- read_csv("hr_train.csv")
test <- read_csv("hr_test.csv")
train <- select(train, -enrollee_id)
test <- select(test, -enrollee_id)

#train$city <- factor(train$city)
train$gender <- factor(train$gender)
train$relevent_experience <- factor(train$relevent_experience)
train$enrolled_university <- factor(train$enrolled_university)
train$education_level <- factor(train$education_level)
train$major_discipline <- factor(train$major_discipline)
train$company_size <- factor(train$company_size)
train$company_type <- factor(train$company_type)
train$last_new_job <- factor(train$last_new_job)
train$target <- factor(train$target)
levels(train$target) <- c(0,1)

#test$city <- factor(test$city)
test$gender <- factor(test$gender)
test$relevent_experience <- factor(test$relevent_experience)
test$enrolled_university <- factor(test$enrolled_university)
test$education_level <- factor(test$education_level)
test$major_discipline <- factor(test$major_discipline)
test$company_size <- factor(test$company_size)
test$company_type <- factor(test$company_type)
test$last_new_job <- factor(test$last_new_job)
levels(test$target) <- c(0,1)



#Since `city_103`, `city_21`, `city_16`, `city 114` and `city_136` have the highest population, I will classify all other cities as `other`.

`%!in%` <- Negate(`%in%`)
train$city = replace(train$city, train$city %!in% c('city_103','city_21',"city_16","city_114","city_160"),"other")

test$city = replace(test$city, test$city %!in% c('city_103','city_21',"city_16","city_114","city_160"),"other")

train$city <- factor(train$city)
test$city <- factor(test$city)


# Logistic Regression
log_model <- glm(target ~., data = train, family = binomial) #Logistic Regression
summary(log_model)


#Errors from the logistic model
log_train_pred <- predict(log_model, train)
train_pred_prob <- binomial()$linkinv(log_train_pred)
train_pred_label <- c(0, 1)[as.factor(train_pred_prob>0.5)]

log_test_pred <- predict(log_model, test)
test_pred_prob <- binomial()$linkinv(log_test_pred)
test_pred_label <- c(0, 1)[as.factor(test_pred_prob>0.5)]

log_train_matrix <- table(train_predicted = train_pred_label, train_actual=train$target)
log_test_matrix <- table(test_predicted = test_pred_label, test_actual=test$target)


#Overall train error
log_train_error <- mean(train_pred_label != train$target)
log_test_error <- mean(test_pred_label != test$target)

log_train_matrix
log_test_matrix
data.frame('target1' = c(log_train_matrix[1,2]/(log_train_matrix[1,2]+log_train_matrix[2,2]),
                         log_test_matrix[1,2]/(log_test_matrix[1,2]+log_test_matrix[2,2])),
           'target0' = c(log_train_matrix[2,1]/(log_train_matrix[2,1]+log_train_matrix[1,1]),
                         log_test_matrix[2,1]/(log_test_matrix[2,1]+log_test_matrix[1,1])),
           Overall = c(log_train_error,log_test_error),
           row.names = c("Logistic_train_error", "Logistic_test_error"))



# Classification tree
cptree <- rpart(target ~ . , train, parms = list(split = "gini"), method = "class")
plotcp(cptree)


#From the plot above, `cp = 0.011` gives the lowest cross-validated error. 
#So, fitting a tree now with `cp = 0.011`.

tree_mod <- rpart(target ~ ., train, parms = list(split = "gini"), method = "class", cp = 0.011)
fancyRpartPlot(tree_mod)

summary(tree_mod)


#Getting errors for tree model
tree_train_pred <- predict(tree_mod, train, type = "class")
tree_train_matrix <- table(train_predicted = tree_train_pred, train_actual = train$target)

tree_test_pred <- predict(tree_mod, test, type = "class")
tree_test_matrix <- table(test_predicted = tree_test_pred, test_actual = test$target)

tree_train_error <- mean(tree_train_pred != train$target)
tree_test_error <- mean(tree_test_pred != test$target)

tree_train_matrix
tree_test_matrix
data.frame(target1 = c(tree_train_matrix[1,2]/(tree_train_matrix[1,2] + tree_train_matrix[2,2]),
                       tree_test_matrix[1,2]/(tree_test_matrix[1,2] + tree_test_matrix[2,2])),
           target0 = c(tree_train_matrix[2,1]/(tree_train_matrix[1,1] + tree_train_matrix[2,1]),
                       tree_test_matrix[2,1]/(tree_test_matrix[1,1] + tree_test_matrix[2,1])),
           Overall = c(tree_train_error, tree_test_error),
           row.names = c("Tree train error", "Tree test error"))



# Random forest
set.seed(1)
rf_mod = randomForest(target ~ ., data = train,
                      importance = TRUE)

rf_train_pred = predict(rf_mod, newdata = train)
rf_train_err = mean(rf_train_pred!=train$target)

rf_test_pred = predict(rf_mod, newdata = test)
rf_test_err = mean(rf_test_pred!=test$target)

rf_train_matrix <- table(train_predicted = rf_train_pred, train_actual = train$target)
rf_test_matrix <- table(test_predicted = rf_test_pred, test_actual = test$target)

data.frame(target1 = c(rf_train_matrix[1,2]/(rf_train_matrix[1,2] + rf_train_matrix[2,2]),
                       rf_test_matrix[1,2]/(rf_test_matrix[1,2] + rf_test_matrix[2,2])),
           target0 = c(rf_train_matrix[2,1]/(rf_train_matrix[1,1] + rf_train_matrix[2,1]),
                       rf_test_matrix[2,1]/(rf_test_matrix[1,1] + rf_test_matrix[2,1])),
           Overall = c(rf_train_err, rf_test_err),
           row.names = c("Random Forest train error", "Random Forest test error"))


#Variable importance by random forest

importance(rf_mod)
varImpPlot(rf_mod, n.var=8, type=1)
varImpPlot(rf_mod, n.var=8, type=2)


# adaBoost

ada_train <- train
ada_train$target <- as.character(train$target)
ada_test <- test
ada_test$target <- as.character(test$target)


set.seed(1)
ada_mod <- gbm(target~., data = ada_train, distribution = "adaboost", n.trees = 5000,
               interaction.depth = 3, shrinkage = 0.1)


#Variable importance using adaBoost
ada_ri <- summary.gbm(ada_mod, plotit = F)$rel.inf
ada_var <- summary.gbm(ada_mod, plotit = F)$var
ada_rel_inf <- data.frame(var=ada_var, rel_inf = ada_ri)
ada_rel_inf <- ada_rel_inf[order(-ada_rel_inf$rel_inf),]
ggplot(data = ada_rel_inf) + geom_bar(aes(x = reorder(ada_var, -ada_ri), y = ada_ri), stat = "identity") + coord_flip() + ylab("relative_influence") + xlab("variables") + theme_classic() + ggtitle("AdaBoost Variable Importance")


#Errors using AdaBoost
ada_train_response <- predict(ada_mod, newdata = train, n.trees = 5000, 
                              type = "response")
ada_train_pred <- ifelse(ada_train_response > 0.5,1,0)
ada_train_err <- mean(ada_train_pred!=train$target)
ada_train_err

ada_test_response <- predict(ada_mod, newdata = test, n.trees = 5000, 
                             type = "response")
ada_test_pred <- ifelse(ada_test_response > 0.5,1,0)
ada_test_err <- mean(ada_test_pred!=test$target)
ada_test_err

ada_train_matrix <- table(train_predicted = ada_train_pred, train_actual = train$target)
ada_test_matrix <- table(test_predicted = ada_test_pred, test_actual = test$target)
ada_train_matrix
ada_test_matrix

data.frame(yes = c(ada_train_matrix[1,2]/(ada_train_matrix[1,2]+ada_train_matrix[2,2]), ada_test_matrix[1,2]/(ada_test_matrix[1,2]+ ada_test_matrix[2,2])),
           no = c(ada_train_matrix[2,1]/(ada_train_matrix[2,1]+ ada_train_matrix[1,1]),
                  ada_test_matrix[2,1]/(ada_test_matrix[2,1]+ ada_test_matrix[1,1])),
           Overall = c(ada_train_err, ada_test_err),
           row.names = c("train error", "test error"))


# Fitting models after smote
s_train <- read.csv("train_after_smote.csv")
s_train <- select(s_train, -X)
s_train$city = replace(s_train$city, s_train$city %!in% c('city_103','city_21',"city_16","city_114","city_160"),"other")

s_train$gender <- factor(s_train$gender)
s_train$relevent_experience <- factor(s_train$relevent_experience)
s_train$enrolled_university <- factor(s_train$enrolled_university)
s_train$education_level <- factor(s_train$education_level)
s_train$major_discipline <- factor(s_train$major_discipline)
s_train$company_size <- factor(s_train$company_size)
s_train$company_type <- factor(s_train$company_type)
s_train$last_new_job <- factor(s_train$last_new_job)
s_train$target <- factor(s_train$target)
s_train$city <- factor(s_train$city)
levels(s_train$target) <- c(0,1)


## Logistic Regression
log_model <- glm(target ~., data = s_train, family = binomial) #Logistic Regression

log_train_pred <- predict(log_model, s_train)
train_pred_prob <- binomial()$linkinv(log_train_pred)
train_pred_label <- c(0, 1)[as.factor(train_pred_prob>0.5)]

log_test_pred <- predict(log_model, test)
test_pred_prob <- binomial()$linkinv(log_test_pred)
test_pred_label <- c(0, 1)[as.factor(test_pred_prob>0.5)]

log_train_matrix <- table(train_predicted = train_pred_label, train_actual=s_train$target)
log_test_matrix <- table(test_predicted = test_pred_label, test_actual=test$target)


#Overall train error
log_train_error <- mean(train_pred_label != train$target)
log_test_error <- mean(test_pred_label != test$target)

log_train_matrix
log_test_matrix
data.frame('target1' = c(log_train_matrix[1,2]/(log_train_matrix[1,2]+log_train_matrix[2,2]),
                         log_test_matrix[1,2]/(log_test_matrix[1,2]+log_test_matrix[2,2])),
           'target0' = c(log_train_matrix[2,1]/(log_train_matrix[2,1]+log_train_matrix[1,1]),
                         log_test_matrix[2,1]/(log_test_matrix[2,1]+log_test_matrix[1,1])),
           Overall = c(log_train_error,log_test_error),
           row.names = c("Logistic_train_error", "Logistic_test_error"))


## Tree
tree_mod <- rpart(target ~ ., s_train, parms = list(split = "gini"), method = "class", cp = 0.011)
tree_train_pred <- predict(tree_mod, s_train, type = "class")
tree_train_matrix <- table(train_predicted = tree_train_pred, train_actual = s_train$target)

tree_test_pred <- predict(tree_mod, test, type = "class")
tree_test_matrix <- table(test_predicted = tree_test_pred, test_actual = test$target)

tree_train_error <- mean(tree_train_pred != train$target)
tree_test_error <- mean(tree_test_pred != test$target)

# tree_train_matrix
# tree_test_matrix
data.frame(target1 = c(tree_train_matrix[1,2]/(tree_train_matrix[1,2] + tree_train_matrix[2,2]),
                       tree_test_matrix[1,2]/(tree_test_matrix[1,2] + tree_test_matrix[2,2])),
           target0 = c(tree_train_matrix[2,1]/(tree_train_matrix[1,1] + tree_train_matrix[2,1]),
                       tree_test_matrix[2,1]/(tree_test_matrix[1,1] + tree_test_matrix[2,1])),
           Overall = c(tree_train_error, tree_test_error),
           row.names = c("Tree train error", "Tree test error"))


## Random Forest
set.seed(1)
rf_mod = randomForest(target ~ ., data = s_train,
                      importance = TRUE)

rf_train_pred = predict(rf_mod, newdata = s_train)
rf_train_err = mean(rf_train_pred!=s_train$target)

rf_test_pred = predict(rf_mod, newdata = test)
rf_test_err = mean(rf_test_pred!=test$target)

rf_train_matrix <- table(train_predicted = rf_train_pred, train_actual = s_train$target)
rf_test_matrix <- table(test_predicted = rf_test_pred, test_actual = test$target)

data.frame(target1 = c(rf_train_matrix[1,2]/(rf_train_matrix[1,2] + rf_train_matrix[2,2]),
                       rf_test_matrix[1,2]/(rf_test_matrix[1,2] + rf_test_matrix[2,2])),
           target0 = c(rf_train_matrix[2,1]/(rf_train_matrix[1,1] + rf_train_matrix[2,1]),
                       rf_test_matrix[2,1]/(rf_test_matrix[1,1] + rf_test_matrix[2,1])),
           Overall = c(rf_train_err, rf_test_err),
           row.names = c("Random Forest train error", "Random Forest test error"))


## AdaBoost
s_ada_train <- s_train
s_ada_train$target <- as.character(s_train$target)
ada_test <- test
ada_test$target <- as.character(test$target)

set.seed(1)
ada_mod <- gbm(target~., data = s_ada_train, distribution = "adaboost", n.trees = 5000,
               interaction.depth = 3, shrinkage = 0.1)

ada_train_response <- predict(ada_mod, newdata = train, n.trees = 5000, 
                              type = "response")
ada_train_pred <- ifelse(ada_train_response > 0.5,1,0)
ada_train_err <- mean(ada_train_pred!=train$target)


ada_test_response <- predict(ada_mod, newdata = test, n.trees = 5000, 
                             type = "response")
ada_test_pred <- ifelse(ada_test_response > 0.5,1,0)
ada_test_err <- mean(ada_test_pred!=test$target)


ada_train_matrix <- table(train_predicted = ada_train_pred, train_actual = train$target)
ada_test_matrix <- table(test_predicted = ada_test_pred, test_actual = test$target)
ada_train_matrix
ada_test_matrix



data.frame(yes = c(ada_train_matrix[1,2]/(ada_train_matrix[1,2]+ada_train_matrix[2,2]), ada_test_matrix[1,2]/(ada_test_matrix[1,2]+ ada_test_matrix[2,2])),
           no = c(ada_train_matrix[2,1]/(ada_train_matrix[2,1]+ ada_train_matrix[1,1]),
                  ada_test_matrix[2,1]/(ada_test_matrix[2,1]+ ada_test_matrix[1,1])),
           Overall = c(ada_train_err, ada_test_err),
           row.names = c("train error", "test error"))


