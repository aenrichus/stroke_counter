library(dplyr)

# script for convolutional network
data <- read.csv("results/conv_test.csv")
data <- mutate(data, distance = abs(target - prediction))
data <- mutate(data, accuracy = ifelse(target == prediction, 1, 0))
data <- mutate(data, network = "convolutional")
data <- mutate(data, type = "test")
test_c <- data

data_acc <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_acc = mean(accuracy))

data_dist <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_dist = mean(distance))

test_conv <- full_join(data_acc, data_dist)
test_conv <- mutate(test_conv, network = "convolutional")
test_conv <- mutate(test_conv, type = "test")

# training
data <- read.csv("results/conv_train.csv")
data <- mutate(data, distance = abs(target - prediction))
data <- mutate(data, accuracy = ifelse(target == prediction, 1, 0))
data <- mutate(data, network = "convolutional")
data <- mutate(data, type = "train")
train_c <- data

data_acc <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_acc = mean(accuracy))

data_dist <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_dist = mean(distance))

train_conv <- full_join(data_acc, data_dist)
train_conv <- mutate(train_conv, network = "convolutional")
train_conv <- mutate(train_conv, type = "train")

data_conv <- full_join(train_c, test_c)
summary_conv <- full_join(train_conv, test_conv)

rm(data, data_acc, data_dist, test_c, test_conv, train_c, train_conv)

# script for feedforward network
data <- read.csv("results/ff_test.csv")
data <- mutate(data, distance = abs(target - prediction))
data <- mutate(data, accuracy = ifelse(target == prediction, 1, 0))
data <- mutate(data, network = "feedforward")
data <- mutate(data, type = "test")
test_f <- data

data_acc <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_acc = mean(accuracy))

data_dist <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_dist = mean(distance))

test_ff <- full_join(data_acc, data_dist)
test_ff <- mutate(test_ff, network = "feedforward")
test_ff <- mutate(test_ff, type = "test")

# training
data <- read.csv("results/ff_train.csv")
data <- mutate(data, distance = abs(target - prediction))
data <- mutate(data, accuracy = ifelse(target == prediction, 1, 0))
data <- mutate(data, network = "feedforward")
data <- mutate(data, type = "train")
train_f <- data

data_acc <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_acc = mean(accuracy))

data_dist <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_dist = mean(distance))

train_ff <- full_join(data_acc, data_dist)
train_ff <- mutate(train_ff, network = "feedforward")
train_ff <- mutate(train_ff, type = "train")

data_ff <- full_join(train_f, test_f)
summary_ff <- full_join(train_ff, test_ff)

rm(data, data_acc, data_dist, test_f, test_ff, train_f, train_ff)

# joining the convolutional and feedforward network data
data_full <- full_join(data_conv, data_ff)
data_summary <- full_join(summary_conv, summary_ff)

write.csv(data_full, "results/kanji_count_5m.csv")
write.csv(data_summary, "results/kanji_count_summary_5m.csv")
