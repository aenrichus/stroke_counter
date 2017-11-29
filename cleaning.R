library(dplyr)

# script for convolutional network
data <- read.csv("summary/test.csv")
data <- mutate(data, distance = abs(target - prediction))
data <- mutate(data, accuracy = ifelse(target == prediction, 1, 0))
data_c <- data
data_c <- mutate(data_c, network = "convolutional")

data_acc <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_acc = mean(accuracy))

data_dist <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_dist = mean(distance))

data_conv <- full_join(data_acc, data_dist)
data_conv <- mutate(data_conv, network = "convolutional")

# script for feedforward network
data <- read.csv("summary_ff/test.csv")
data <- mutate(data, distance = abs(target - prediction))
data <- mutate(data, accuracy = ifelse(target == prediction, 1, 0))
data_f <- data
data_f <- mutate(data_f, network = "feedforward")

data_acc <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_acc = mean(accuracy))

data_dist <- data %>% 
  group_by(trials, target) %>% 
  summarise(mean_dist = mean(distance))

data_ff <- full_join(data_acc, data_dist)
data_ff <- mutate(data_ff, network = "feedforward")

# joining the convolutional and feedforward network data
data_full <- full_join(data_c, data_f)
data_summary <- full_join(data_conv, data_ff)

write.csv(data_full, "kanji_count.csv")
write.csv(data_summary, "kanji_count_summary.csv")
