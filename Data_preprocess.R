hr_data = read.csv('aug_train.csv')

# Dealing with NA values and type:

#There are a lot of missing values in almost all the columns, directly drop them would lose too much observations.
#So I deal with them individually based on the meaning of that column,
# then cast that column to the type that corresponding to its meaning.

hr_data$enrollee_id = as.numeric(hr_data$enrollee_id) # numeric type

hr_data$city = as.factor(hr_data$city) # factor type

hr_data$city_development_index = as.numeric(hr_data$city_development_index) # numeric type

unique(hr_data$gender)
# There are a lot of missing values in column gender, I classify all the missing values to the existing category "Other".
hr_data$gender<-replace(hr_data$gender, hr_data$gender=="", "Other")
hr_data$gender = as.factor(hr_data$gender) # factor type

hr_data$relevent_experience = as.factor(hr_data$relevent_experience) # factor type

unique(hr_data$enrolled_university)
# There are a lot of missing values in column enrolled_university,
# I change all the missing values to "Unknown", a new category.
hr_data$enrolled_university<-replace(hr_data$enrolled_university, hr_data$enrolled_university=="", "Unknown")
hr_data$enrolled_university = as.factor(hr_data$enrolled_university) # factor type

unique(hr_data$education_level)
# There are a lot of missing values in column education_level,
# I change all the missing values to "Unknown", a new category.
hr_data$education_level<-replace(hr_data$education_level, hr_data$education_level=="", "Unknown")
hr_data$education_level = as.factor(hr_data$education_level) # factor type

unique(hr_data$major_discipline)
# There are missing values, "No Major" and "Other" in column major_discipline,
# I change all of them to "Other" class.
hr_data$major_discipline<-replace(hr_data$major_discipline, hr_data$major_discipline=="", "Other")
hr_data$major_discipline<-replace(hr_data$major_discipline, hr_data$major_discipline=="No Major", "Other")
hr_data$major_discipline = as.factor(hr_data$major_discipline) # factor type

unique(hr_data$experience)
hr_data$experience<-replace(hr_data$experience, hr_data$experience==">20", "20")
hr_data$experience<-replace(hr_data$experience, hr_data$experience=="<1", "0")
hr_data$experience<-replace(hr_data$experience, hr_data$experience=="", "0")
hr_data$experience = as.numeric(hr_data$experience) # numeric type

unique(hr_data$company_size)
hr_data$company_size<-replace(hr_data$company_size, hr_data$company_size=="", "<10")
hr_data$company_size<-replace(hr_data$company_size, hr_data$company_size=="10/49", "10-49")
hr_data$company_size = as.factor(hr_data$company_size) # factor type

unique(hr_data$company_type)
hr_data$company_type<-replace(hr_data$company_type, hr_data$company_type=="", "Other")
hr_data$company_type = as.factor(hr_data$company_type) # factor type

unique(hr_data$last_new_job)
hr_data$last_new_job<-replace(hr_data$last_new_job, hr_data$last_new_job=="", "Unknown")
hr_data$last_new_job = as.factor(hr_data$last_new_job) # factor type

unique(hr_data$training_hours)
hr_data$training_hours = as.numeric(hr_data$training_hours) # numeric type

hr_data$target = as.numeric(hr_data$target) # factor type

# Train Test split

nrow(hr_data)
# 19158 obs with label.

hr_data_yes = hr_data[hr_data$target == 1,]
nrow(hr_data_yes)
# 4777 obs with target == 1.

hr_data_no = hr_data[hr_data$target == 0,]
nrow(hr_data_no)
# 14381 obs with target == 0.

# Split 20% of each class to form the test set and use the rest 80% of each class to form the training set.

hr_test_yes_index = sample(1:4777, 955, F)
hr_test_no_index = sample(1:14381, 2876, F)
hr_test_yes = hr_data_yes[hr_test_yes_index,]
hr_test_no = hr_data_no[hr_test_no_index,]
hr_test = rbind(hr_test_yes, hr_test_no)
hr_test = hr_test[sample(1:nrow(hr_test)),]
#hr_test has 3831 obs in total
#The ratio of positive and negative samples is same to the original data set.
#No undersampling or oversampling here, you should adjust the sample ratio based on your own model property.
write.csv(hr_test,"hr_test.csv", row.names = FALSE)
#Some of the column type might switch to character from factor after the write and read process,
#If you need factor as your input, you should cast again manually in your R script.

hr_train_yes = hr_data_yes[-hr_test_yes_index,]
hr_train_no = hr_data_no[-hr_test_no_index,]
hr_train = rbind(hr_train_yes, hr_train_no)
hr_train = hr_train[sample(1:nrow(hr_train)),]
#hr_train has 15327 obs in total
#The ratio of positive and negative samples is same to the original data set.
#No undersampling or oversampling here, you should adjust the sample ratio based on your own model property.
write.csv(hr_train,"hr_train.csv", row.names = FALSE)
#Some of the column type might switch to character from factor after the write and read process,
#If you need factor as your input, you should cast again manually in your R script.


