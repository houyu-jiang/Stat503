library(tidyverse)
library(ggmosaic)
library(GGally)
library(reshape)
library(summarytools)
# original dataset
train = read.csv('aug_train.csv', na.strings = c('',' '))
train_cls0 = train %>% filter(target == 0)
train_cls1 = train %>% filter(target == 1)
cls = ifelse(train$target==0, 'No', 'Yes')
perc = train %>% 
  mutate(Class=factor(cls, labels = c('class 0', 'class 1'))) %>%
  group_by(Class) %>%
  count() %>% 
  ungroup %>%
  mutate(perc=`n`/sum(`n`)) %>%
  arrange(perc) %>%
  mutate(labels = scales::percent(perc))
ggplot(data=perc, aes(x='', y=perc, fill=Class))+geom_col()+
  geom_text(aes(label=labels), position = position_stack(vjust = 0.5))+
  labs(x=NULL, y=NULL, title = 'Imbalance between Classes')+
  coord_polar(theta='y')
# numerical 
summary(train[, sapply(train, class) %in% c('numeric', 'integer', 'double')])
cls = ifelse(train$target==0, 'No', 'Yes')
train_boxplot = train %>% 
  mutate(cls = cls) %>%
  dplyr::select(city_development_index, training_hours, cls) %>%
  gather(key='variables', value='value', -cls) %>%
  mutate(cls=factor(cls, labels = c('class 0', 'class 1')))
ggplot(train_boxplot) +
  geom_boxplot(aes(x=cls, y=value)) +
  facet_wrap(~variables, scales='free', nrow=1) +
  labs(x='Candidate Type') +
  theme_bw()
# city
train1 = train
train1$city = as.factor(train$city)
city.perc = train1 %>% 
  group_by(city) %>%
  count() %>% 
  ungroup %>%
  mutate(perc=`n`/sum(`n`)) %>%
  arrange(perc) %>%
  mutate(labels = scales::percent(perc))
city.perc = rbind(city.perc, data.frame(city='others', n=7370, perc=0.38469569, labels='38.4686%'))
city.perc %>% arrange(desc(perc)) %>%
  head(8) %>%
  ggplot(aes(x='', y=perc, fill=city))+geom_col()+
  geom_text(aes(label=labels), position = position_stack(vjust = 0.5))+
  labs(x=NULL, y=NULL, title = 'City Percentage')+
  coord_polar(theta='y')
# Subset character columns
df_char = train[, sapply(train, class) %in% 
                  c('character', 'factor')]
dfSummary(df_char)
train1 = train
train1$company_type = as.factor(train$company_type)
train1 %>% 
  group_by(company_type) %>%
  summarise(count = n()) %>%
  ggplot(aes(x=company_type, y=count))+
  geom_bar(stat = "identity")+
  theme_bw() 
train1$company_size = as.factor(train$company_size)
train1 %>% 
  group_by(company_size) %>%
  summarise(count = n()) %>%
  ggplot(aes(x=company_size, y=count))+
  geom_bar(stat = "identity")+
  theme_bw() 
# education_level
train_panel = train %>% 
  mutate(cls = cls) %>%
  dplyr::select(education_level, cls) %>%
  mutate(cls=factor(cls, labels = c('class 0', 'class 1')))
ggplot(train_panel) + 
  geom_mosaic(aes(x=product(cls), fill=education_level)) +
  labs(fill='Education level', y='Levels', x='Candidate Type')
# pair plot
train_pairs = train %>% mutate(cls = cls) 
ggpairs(train_pairs, columns = c(3,13), 
        mapping=aes(color=factor(cls, labels = c('class 0', 'class 1'))),
        diag = list(continuous = wrap("densityDiag", alpha = .5)),
        title = "Pair Plot of Job Dataset",
        legend = 1) +
  labs(fill = "Candidate Type")+
  theme_bw() +
  theme(plot.title = element_text(size = 15, hjust = 0.5))
# missing value analysis
na_target0 = colSums(is.na(train_cls0))
na_target1 = colSums(is.na(train_cls1))

# Plot for missing values
class_0 = na_target0 / dim(train_cls0)[1] * 100
class_1 = na_target1 / dim(train_cls1)[1] * 100
df_na = data.frame(class_0, class_1)
df_na= rownames_to_column(as.data.frame(t(df_na)))
names(df_na) = c('class', 'id', 'city', 'city_index', 'gender', 
                 'experience', 'university', 'edu_level', 'major',
                 'exp_years', 'comp_size', 'comp_type', 'last_job',
                 'train_hours', 'target')
df_na.m = melt(df_na, id.vars = 'class') %>% 
  dplyr::select(variable, class, value)
ggplot(df_na.m, aes(variable, value) ) +
  geom_bar(aes(fill = class), width = 0.4, 
           position = "dodge", stat="identity") +
  labs(y = 'Missing Percentage') + 
  theme_bw() 




















