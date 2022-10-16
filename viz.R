
library(data.table)
library(magrittr)
library(ggplot2)

setwd('~/School/assignment_2/')

df = fread('20221014-rand-opt159.csv')

df[fitness=='OneMax'] %>%
  ggplot(aes(x=input_size, y = best_fitness, color = algorithm)) +
  geom_point() +
  geom_smooth(method='lm')

### OneMax iterations much higher for random and simulated annealing
### FlopFlop iterations much higher for simulated annealing
### 4 Peaks iterations much higher for simulated annealing

### 4 peaks mimic and genetit outperform simluated annealing and much
# outperform random
### Flip Flop performance is pretty similar
### OneMax too

# mimic performs best on fourpeaks

df[fitness=='FourPeaks'] %>%
  ggplot(aes(x=input_size, y = best_fitness, color = algorithm)) +
  geom_point(aes(shape=algorithm)) +
  # geom_smooth(method='lm')
  geom_line()

## simulated annealing
(
  plot = df[fitness=='FlipFlop'] %>%
    ggplot(aes(x = number_of_iterations, y = best_fitness)) +
    geom_point(aes(color = input_size)) +
    facet_wrap(~algorithm) +
    theme_minimal() +
    labs(
      x = 'Number of Iterations',
      y = 'Best Fitness Score',
      color = 'Input Size*',
      caption = '* number of potential items to choose from',
      title = 'FlipFlop Problem: best fitness score by number of algorithm iterations',
      subtitle = 'faceted by algorithm'
    )
)

ggsave('plots/flipflop.png', width=8, height=4, dpi=300)

(
  plot = df[fitness=='FlipFlop'] %>%
    ggplot(aes(x = input_size, y = number_of_iterations)) +
    # geom_point(aes(color = input_size)) +
    geom_point() +
    facet_wrap(~algorithm) +
    theme_minimal() +
    labs(
      y = 'Number of Iterations',
      x = 'Input Length',
      title = 'FlipFlop Problem: number of iterations by input length',
      subtitle = 'faceted by algorithm'
    )
)

ggsave('plots/flipflop2.png', width=8, height=4, dpi=300)


## MIMIC
(
  plot = df[fitness=='FourPeaks'] %>%
    ggplot(aes(x = number_of_iterations, y = best_fitness)) +
    geom_point(aes(color = input_size)) +
    facet_wrap(~algorithm) +
    theme_minimal() +
    scale_x_log10() +
    labs(
      x = 'Number of Iterations (log10)',
      y = 'Best Fitness Score',
      color = 'Input Size*',
      caption = '* number of potential items to choose from',
      title = 'FourPeaks Problem: best fitness score by number of algorithm iterations',
      subtitle = 'faceted by algorithm'
    )
)

ggsave('plots/4peaks.png', width=8, height=4, dpi=300)

(
  plot = df[fitness=='FourPeaks'] %>%
    ggplot(aes(x = input_size, y = time)) +
    geom_point() +
    facet_wrap(~algorithm) +
    theme_minimal() +
    labs(
      x = 'Input Size',
      y = 'Time to Complete',
      title = 'FourPeaks Problem: time to complete by input size',
      subtitle = 'faceted by algorithm'
    )
)

ggsave('plots/4peaks2.png', width=8, height=4, dpi=300)

### KNAPSACK
bag = fread('20221015-knapsack-131.csv')

bag %>%
  ggplot(aes(x=input_size, y = time, color = algorithm)) +
  geom_point(aes(shape=algorithm)) +
  # geom_smooth(method='lm')
  geom_line()

(
plot = bag %>%
  ggplot(aes(x = number_of_iterations, y = best_fitness)) +
  geom_point(aes(color = input_size)) +
  facet_wrap(~algorithm) +
  theme_minimal() +
  scale_x_log10() +
  labs(
    x = 'Number of Iterations (log10 scale)',
    y = 'Best Fitness Score',
    color = 'Input Size*',
    caption = '* number of potential items to choose from',
    title = 'Knapsack Problem: best fitness score by number of algorithm iterations',
    subtitle = 'faceted by algorithm'
  )
)

ggsave('plots/knapsack.png', width=8, height=4, dpi=300)


(
  plot = bag %>%
    ggplot(aes(x = time, y = best_fitness)) +
    geom_point(aes(color = input_size)) +
    facet_wrap(~algorithm) +
    theme_minimal() +
    # scale_x_log10() +
    labs(
      x = 'Time to complete (seconds)',
      y = 'Best Fitness Score',
      color = 'Input Size*',
      caption = '* number of potential items to choose from',
      title = 'Knapsack Problem: best fitness score by time to complete',
      subtitle = 'faceted by algorithm'
    )
)

ggsave('plots/knapsack2.png', width=8, height=4, dpi=300)

######## NN

nn = fread('20221015-nn.csv')
nn$fitness_curve = NULL

(
  plot = nn[hidden_nodes=="[8]"] %>%
    ggplot(aes(x=sample_size)) +
    geom_point(aes(y=train_accuracy, color='train_accuracy'), size=0.5) +
    geom_point(aes(y=test_accuracy, color='test_accuracy'), size=0.5) +
    geom_smooth(aes(y=train_accuracy, color='train_accuracy'), alpha=0.2) +
    geom_smooth(aes(y=test_accuracy, color='test_accuracy'), alpha=0.2) +
    facet_wrap(~algorithm) +
    theme_minimal() +
    labs(
      x = 'Sample Size',
      y = 'Accuracy',
      color = NULL,
      title = 'Neural Network: accuracy by sample size',
      subtitle = 'faceted by algorithm'
    )
)

ggsave('plots/nn1-accuracy.png', height=8, width=4, dpi=300)

(
  plot = nn[hidden_nodes=="[8]"] %>%
    ggplot(aes(x=sample_size)) +
    geom_point(aes(y=f1_train, color='train f1'), size=0.5) +
    geom_point(aes(y=f1_test, color='test f1'), size=0.5) +
    geom_smooth(aes(y=f1_train, color='train f1'), alpha=0.2) +
    geom_smooth(aes(y=f1_test, color='test f1'), alpha=0.2) +
    facet_wrap(~algorithm) +
    theme_minimal() +
    labs(
      x = 'Sample Size',
      y = 'f1-score',
      color = NULL,
      title = 'Neural Network: f1 score by sample size',
      subtitle = 'faceted by algorithm'
    )
)

ggsave('plots/nn1-f1.png', height=8, width=4, dpi=300)

(
  plot = nn[hidden_nodes == "[8]" & sample_size == 1932] %>%
    ggplot(aes(x = train_accuracy, fill = algorithm)) +
    geom_histogram(show.legend = F) +
    facet_wrap(~algorithm) +
    theme_minimal() +
    labs(
      x = 'Training Accuracy',
      y = 'Count',
      title = 'Training Accuracy Distirbution over numerous independent model fits',
      subtitle='faceted by algorithm'
    )
)

ggsave('plots/nn1-accuracy-dist.png', height=8, width=4, dpi=300)


nn[hidden_nodes == "[8]" & sample_size == 1932][order(-f1_train)][1]

"
best: genetic algorithm

train_accuracy test_accuracy    f1_train    f1_test
0.9415114     0.9378882 0.763102725 0.73214286
"

# nn = fread('20221015-nn-2.csv')
# 
# nn[hidden_nodes == "[8]" & sample_size == 1932][order(-f1_train)][1]
# 
# 
# nn[hidden_nodes == "[8]" & max_iters==400] %>%
#   ggplot(aes(x = sample_size)) +
#   geom_point(aes(y=train_accuracy, color='train_accuracy'), size=0.5) +
#   geom_point(aes(y=test_accuracy, color='test_accuracy'), size=0.5) +
#   geom_smooth(aes(y=train_accuracy, color='train_accuracy'), alpha=0.2) +
#   geom_smooth(aes(y=test_accuracy, color='test_accuracy'), alpha=0.2) +
#   facet_wrap(~algorithm) +
#   theme_minimal()


