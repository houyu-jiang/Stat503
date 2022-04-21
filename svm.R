library(tidyverse)

train_data = read.csv('hr_train.csv') %>% select(-enrollee_id)
test_data = read.csv('hr_test.csv') %>% select(-enrollee_id)
train_x = train_data[,-13]
train_y = train_data[,13]
test_x = test_data[,-13]
test_y = test_data[,13]

library(lattice)
library(caret)
train_x = train_x %>% mutate_if(is.numeric, scale)
test_x = test_x %>% mutate_if(is.numeric, scale)

`%!in%` <- Negate(`%in%`)
train_x1 = train_x
train_x1$city = replace(train_x1$city, train_x1$city %!in% 
                          c('city_103', 'city_21', 'city_16', 'city_114', 'city_160'), 'other')
test_x1 = test_x
test_x1$city = replace(test_x1$city, test_x1$city %!in% 
                 c('city_103', 'city_21', 'city_16', 'city_114', 'city_160'), 'other')
dummies = dummyVars(~., data=train_x1)
c2 = predict(dummies, train_x1)
new_train_data = as.data.frame(cbind(c2, train_y))
dummies = dummyVars(~., data=test_x1)
c2 = predict(dummies, test_x1)
new_test_data = as.data.frame(cbind(c2, test_y))

# +

library(e1071)
gammalist <- c(0.01,0.05,0.1,0.5,1,5,7,10,20,50,100)
costlist = c(0.01,0.1,1,10,20,50)
# -

tune.out = tune.svm(as.factor(train_y) ~., data=new_train_data, kernel='radial', 
                    cost=0.01, gamma=gammalist)
summary(tune.out)
summary(tune.out$best.model)
svm1 = predict(tune.out$best.model, new_test_data[,-ncol(new_test_data)])
cm1 = confusionMatrix(svm1, as.factor(new_test_data$test_y))

cm1

svm1.1 = predict(tune.out$best.model, new_train_data[,-ncol(new_train_data)])

confusionMatrix(svm1.1, as.factor(new_train_data$train_y))

summary(tune.out)

tune.out2 = tune.svm(as.factor(train_y) ~., data=new_train_data, kernel='radial', 
                    cost=0.1, gamma=gammalist)
summary(tune.out2)
summary(tune.out2$best.model)
svm2 = predict(tune.out2$best.model, new_test_data[,-ncol(new_test_data)])
cm2 = confusionMatrix(svm2, as.factor(new_test_data$test_y))

summary(tune.out2)
cm2

tune.out3 = tune.svm(as.factor(train_y) ~., data=new_train_data, kernel='radial', 
                    cost=1, gamma=gammalist)
summary(tune.out3)
summary(tune.out3$best.model)
svm3 = predict(tune.out3$best.model, new_test_data[,-ncol(new_test_data)])
cm3 = confusionMatrix(svm3, as.factor(new_test_data$test_y))

summary(tune.out3)
cm3

# best one
tune.out4 = tune.svm(as.factor(train_y) ~., data=new_train_data, kernel='radial', 
                    cost=5, gamma=gammalist)
summary(tune.out4)
summary(tune.out4$best.model)
svm4 = predict(tune.out4$best.model, new_test_data[,-ncol(new_test_data)])
cm4 = confusionMatrix(svm4, as.factor(new_test_data$test_y))

summary(tune.out4)
cm4

svm4.1 = predict(tune.out4$best.model, new_train_data[,-ncol(new_train_data)])

confusionMatrix(svm4.1, as.factor(new_train_data$train_y))

tune.out5 = tune.svm(as.factor(train_y) ~., data=new_train_data, kernel='radial', 
                    cost=10, gamma=gammalist)
summary(tune.out5)
summary(tune.out5$best.model)
svm5 = predict(tune.out5$best.model, new_test_data[,-ncol(new_test_data)])
cm5 = confusionMatrix(svm5, as.factor(new_test_data$test_y))

summary(tune.out5)
cm5

tune.out6 = tune.svm(as.factor(train_y) ~., data=new_train_data, kernel='radial', 
                    cost=50, gamma=gammalist)
summary(tune.out6)
summary(tune.out6$best.model)
svm6 = predict(tune.out6$best.model, new_test_data[,-ncol(new_test_data)])
cm6 = confusionMatrix(svm6, as.factor(new_test_data$test_y))

summary(tune.out6)
cm6

# Oversampling for imbalanced classes

library(smotefamily)
smote = SMOTE(new_train_data[,-13], new_train_data$train_y)
new = smote$data

new$class <- as.numeric(new$class)
table(new$class)

new1 = new %>% select(-experience, -training_hours, -train_y, -class, -city_development_index)

m = dim(new1)[2]
for(i in 1:m){
    new1[,i] = ifelse(new1[,i]>0.5, 1,0)
}

new1$experience = new$experience
new1$training_hours = new$training_hours
new1$city_development_index = new$city_development_index
new1$train_y = new$train_y

svm.smote = svm(as.factor(train_y) ~., data=new1, kernel='radial', 
                    cost=5, gamma=0.01)
svm7.1 = predict(svm.smote, new1[,-ncol(new1)])
svm7 = predict(svm.smote, new_test_data[,-ncol(new_test_data)])
cm7 = confusionMatrix(svm7, as.factor(new_test_data$test_y))
cm7.1 = confusionMatrix(svm7.1, as.factor(new1$train_y))

cm7.1

cm7
