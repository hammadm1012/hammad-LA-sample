## Mariners R&D Problem Set
## By Mohamed Hammad / hammadm1012@gmail.com

# Below is my code for coding challenge #1. 
# I have also seperately provided 'Challenge #1 Brief - Mohamed Hammad' discussing my process and the 
# steps I would take for improvement. 

# This needs only the two files 2020-train and 2020-test in the same working directory
# along with the packages below


# libraries
library(tidyverse)
library(caret)
library(ranger)
library(e1071)
library(doParallel)

# configure multicore for later work
n_use <- detectCores()/2
cl <- makeCluster(n_use)
registerDoParallel(cl)


# loading data provided
train_data <- read.csv(file = "2020-train.csv")
test_data <- read.csv(file = "2020-test.csv")

# defining is_strike in train_data and having been tasked with predicting whether a pitch will 
# be a strike *of any kind* is_strike will return 1 for any of the following pitch_call:
# 'FoulBall', 'InPlay', 'StrikeCalled', 'StrikeSwinging'. Else 0. 
train_data$is_strike <- ifelse(train_data$pitch_call == "StrikeCalled" | 
                        train_data$pitch_call == "InPlay" | train_data$pitch_call == "FoulBall" |
                       train_data$pitch_call == "StrikeSwinging", 1, 0)

#to avoid issues with ranger and the random forest model
train_data$is_strike <- as.factor(train_data$is_strike) 

# removing the pitch_id and pitch_call columns from training data
train_data <- train_data[, -c(35, 36)]

# removing tilt column from both dataframes
train_data <- train_data[, -c(18)]
test_data <- test_data[, -c(18)]

# checking our train_data for percent of incomplete rows
sum(is.na(train_data))/nrow(train_data)

# removing incomplete rows from our train_data because they stand at 0.01% of the set
train_data <- na.omit(train_data)

# reviewing levels
sapply(train_data, levels)

# because of categorical limits to randomForest changing some columns in our train_data to
# character from factor and the same for test_data specifically; pitcher_id, batter_id, stadium_id,
# umpire_id, catcher_id
train_data$pitcher_id <- as.character(train_data$pitcher_id)
train_data$batter_id <- as.character(train_data$batter_id)
train_data$stadium_id <- as.character(train_data$stadium_id)
train_data$umpire_id <- as.character(train_data$umpire_id)
train_data$catcher_id <- as.character(train_data$catcher_id)

train_data$pitcher_id <- as.character(train_data$pitcher_id)
train_data$batter_id <- as.character(train_data$batter_id)
train_data$stadium_id <- as.character(train_data$stadium_id)
train_data$umpire_id <- as.character(train_data$umpire_id)
train_data$catcher_id <- as.character(train_data$catcher_id)

# fixing the pitcher_side and batter_side for uniformity by replacing the ''L' and 'R' with 'Left' and 'Right'
train_data$pitcher_side[train_data$pitcher_side == 'L'] <- 'Left'
train_data$pitcher_side[train_data$pitcher_side == 'R'] <- 'Right'
train_data$pitcher_side <- droplevels(train_data$pitcher_side) #dropping unused levels 'L' and 'R'
train_data$batter_side[train_data$batter_side == 'L'] <- 'Left'
train_data$batter_side[train_data$batter_side == 'R'] <- 'R'
train_data$batter_side <- droplevels(train_data$batter_side) 

#the same for test_data
test_data$pitcher_side[test_data$pitcher_side == 'L'] <- 'Left'
test_data$pitcher_side[test_data$pitcher_side == 'R'] <- 'Right'
test_data$pitcher_side <- droplevels(test_data$pitcher_side) #dropping unused levels 'L' and 'R'
test_data$batter_side[test_data$batter_side == 'L'] <- 'Left'
test_data$batter_side[test_data$batter_side == 'R'] <- 'R'
test_data$batter_side <- droplevels(test_data$batter_side) 

# removing y55 column from both dataframes because of no variation
train_data <- train_data[, -c(31)]
test_data <- test_data[, -c(31)]


# now with ready data making a random forest model using ranger

rf_model <- ranger(is_strike ~ ., data = train_data, importance = "impurity",
                   num.trees = 3000, seed = 101, verbose = TRUE)

# checking prediction error which turns out to be 13%
rf_model$prediction.error 

# checking  for personal observation the variable importance in our model
rf_model$variable.importance

sum(is.na(test_data))/nrow(test_data) # checking for incomplete rows in test_data

# will just drop incomplete rows because of limitations imputing
rm_test_cols <- colnames(test_data[, -c(33)])
rm_test_cols # print names to copy below
# omitting tbose incomplete rows excluding is_strike
test_data <- test_data %>% drop_na("pitcher_id", "pitcher_side", "batter_id", "batter_side",
                                    "stadium_id", "umpire_id", "catcher_id", "inning", "top_bottom", 
                                    "outs", "balls", "strikes", "release_speed", "vert_release_angle",
                                    "horz_release_angle", "spin_rate","spin_axis","rel_height",
                                    "rel_side","extension","vert_break","induced_vert_break","horz_break",
                                    "plate_height","plate_side","zone_speed","vert_approach_angle",
                                    "horz_approach_angle","zone_time","x55","z55","pitch_type",
                                    "pitch_id")

sum(is.na(test_data))/nrow(test_data) # checking again incomplete rows in test_data

# predicting is_strike in test_data with ranger
# excluding the pitch_id from the test_data for prediction purposes 
test_data$is_strike <- predict(rf_model, data = test_data[, -c(34)], predict.all = FALSE, 
        num.trees = rf_model$num.trees, type = "response", seed = 101, verbose = TRUE)$predictions


# writing out submitted csvs with the full test_data as one and the other a frame with only
# pitch_id and is_strike
write.csv(test_data, file = "completed_test_data.csv", row.names = FALSE)
id_strike <- subset(test_data, select = c(33, 34))
write.csv(id_strike, file = "pitch_id_and_strike.csv", row.names = FALSE)

# STOP CLUSTER
stopCluster(cl)



