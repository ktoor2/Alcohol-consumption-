# ========================================
# Author: Kunwar Singh
# Project: Individual Project for Data Science Course
# Dataset: Alcohol comsuption among teens
# Problem Type: Classification 
#========================================

library(dplyr)
library(ggplot2)
library(gridExtra)
library(grid)
library(glmulti)
library(bestglm)
library(MASS)
library(car)
library(stringi)
library(ResourceSelection)
library(e1071)
library(ROCR)
library(caret)
library(pROC)
library(boot)
library(waffle)


data_set <- read.table(file="C:\\Users\\kunwar\\Downloads\\student (1)\\student-por.csv",sep = ";",header = TRUE)
data_set_test <-read.table(file="C:\\Users\\kunwar\\Downloads\\student (1)\\student-mat.csv",sep = ";",header = TRUE)
View(data_set)

waffle.col <- c("#00d27f","#adff00","#f9d62e","#fc913a","#ff4e50")


#====== Exploratory data analysis======

#distribution of alcohol comsuption among teens
Dalc_factor <- factor(data_set$Dalc,ordered = TRUE)
Walc_factor <- factor(data_set$Walc,ordered = TRUE)
health_factor <- factor(data_set$health,ordered = TRUE)

 

p1 <- ggplot(data_set, aes(x = sex, y = age, fill = sex)) + geom_boxplot() +
  facet_wrap(~ Dalc, ncol = 5)+ ggtitle("Weekday alcohol consumption")
p2 <-ggplot(data_set, aes(x = sex, y = age, fill = sex)) + geom_boxplot() +
  facet_wrap(~ Walc, ncol = 5) + ggtitle("Weekend alcohol consumption")
grid.arrange(p1,p2, ncol = 2)


#affect of family relationships to alcohol comsuption


weekday_graphs <- list()
weekend_graphs <- list()

for (i in 1:5)
{
  fam_data <- filter(data_set,data_set$famrel == i)
  weekday_graphs[[i]] <- qplot (Dalc,data = fam_data, ylim = c(0,150), xlab = "Alcohol consumption", binwidth = 1)
  weekend_graphs[[i]] <- qplot (Walc,data = fam_data, ylim = c(0,150), xlab = "Alcohol consumption", main = i)
}
do.call(grid.arrange, weekend_graphs)

#distribution of alcohol consumption overall
qplot(Dalc_factor)
qplot(Walc_factor)

#Final Grades w.r.t alcohol consumption
ggplot(data_set, aes(x = sex, y = G3, fill = sex)) + geom_boxplot() +
  facet_wrap(~ Dalc, ncol = 5)

#alcohol consumption and absences 

waffle.col <- c("#00d27f","#adff00","#f9d62e","#fc913a","#ff4e50")

#k means clustering of G1 and G2

set.seed(20)
Cluster <- kmeans(data_set[,c(31,32) ], 5,  nstart = 20)
Cluster
Cluster$cluster <- as.factor(Cluster$cluster)
ggplot(data_set, aes(data_set$G1, data_set$G2, color = Cluster$cluster)) + geom_point()

ggplot(data.source, aes(x=Dalc, y=absences, fill=Dalc))+
  geom_violin()+
  scale_fill_manual(values = waffle.col)+
  theme_bw()+
  theme(legend.position="none")+
  ggtitle("Absences distribution per Workday alcohol consumption")+
  xlab("Alcohol consumption")+
  ylab("Number of school absences")




#=======Finding the best predictors=========

#Removing some columns that do not uphold for our linear model
data_set_new<- subset(data_set, select = -c(G1,G2) )

#if alc <=2 then alcohol consumption is average/no else it is high/yes
for(i in 1:nrow(data_set_new))
{
  if(data_set_new$Walc[i]  <= 2)
  {
    data_set_new$Walc[i] <- 0
  }
  else
  {
    data_set_new$Walc[i] <- 1
  }
}
#data_set_new$Walc <- as.factor(data_set_new$Walc)

View(data_set_new)
## Stepwise Regression

fit <- glm(data_set_new$Walc~.,data=data_set_new,family = binomial)
step <- stepAIC(fit, direction="both")
step$anova # display results




mat_subset <- regsubsets(Walc ~ ., data = data_set_new,nbest=1)
summary(mat_subset)

dframe <- data.frame(est = c(summary(mat_subset)$adjr2, 
                             summary(mat_subset)$cp,
                             summary(mat_subset)$bic),
                     x = rep(1:8, 33),
                     type = rep(c("adjr2", "cp", "bic"), 
                                each = 8))
qplot(x, est, data = dframe, geom = "line") +
  theme_bw() + facet_grid(type ~ ., scales = "free_y")

#Based of forward and backword stepwise model and subset models was found to be
#sex + Fedu + Fjob + studytime + famrel + 
 # goout + Dalc + health


model1 <- glm(Walc~sex + Fedu + Fjob + studytime + famrel + 
                goout + Dalc + health,data=data_set_new, family=binomial)
summary(model1)
model2 <- glm(Walc~health+Dalc+goout+studytime+sex, data = data_set_new, family = binomial)
summary(model2)

anova(model1,model2, test = "Chisq")




#======Model analysis================


summary(model1)
 
model1
#now lets do some classification prediction
predict(model1,data.frame(health=5,goout=5, sex='M',studytime=2,Dalc = 1,Pstatus='T',reason='course',
                              schoolsup='yes',activities='yes',famrel=1,freetime=3, type='Yes',
                          Fedu=4,Fjob='teacher'),type = 'response')




pred_prob <- predict(model1,data_set_new,type = 'response')
pred <- prediction(pred_prob,data_set_new$Walc)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))

  
pred_prediction <- rep(0,649)
pred_prediction
pred_prediction[pred_prob > 0.5] = 1
pred_prediction

# confustion matrix on training data
confusionMatrix(table(pred_prediction,data_set_new$Walc))

#confusion matrix on test data
View(data_set_test)
data_set_test<- subset(data_set_test, select = -c(G1,G2) )
for(i in 1:nrow(data_set_test))
{
  if(data_set_test$Walc[i]  <= 2)
  {
    data_set_test$Walc[i] <- 0
  }
  else
  {
    data_set_test$Walc[i] <- 1
  }
}

pred_prob_test <- predict(model1,data_set_test,type = 'response')
pred_prediction_test <- rep(0,395)
pred_prediction_test[pred_prob_test > 0.5] = 1



pred <- prediction(pred_prob_test,data_set_test$Walc)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))



length(pred_prediction_test)
length(data_set_test$Walc)
confusionMatrix(table(pred_prediction_test,data_set_test$Walc))

#========= applyin k-fold validation=========#

set.seed(1000)
crsv_10 <- cv.glm(data = data_set_new, glmfit = model1 , K = 10)
crsv_10$delta[1]
#we can also see how the model performs with model2
set.seed(1000)
crsv_10_m2 <- cv.glm(data = data_set_test, glmfit = model2 , K = 10)
crsv_10_m2$delta[1]
 
#so we can see here that model1 is better