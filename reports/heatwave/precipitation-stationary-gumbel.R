require(UNSEEN)
require(ggplot2)
require(extRemes)

SEAS5_events <- read.csv("../Data/drought/v3/SEAS5_events.csv")

SEAS5_events$year <- ymd(paste0(SEAS5_events$year, "-01-01"))

SEAS5_events$tprate_transformed <- 1 / SEAS5_events$tprate
print(head(SEAS5_events))

UNSEEN_1 <- SEAS5_events[SEAS5_events$number < 25 & 
                           SEAS5_events$year < '1984-02-01' &
                           SEAS5_events$year > '1981-02-01',]

UNSEEN_2 <- SEAS5_events[SEAS5_events$number < 25 & 
                           SEAS5_events$year > '2020-01-01',]

gev_fit_1 <- fevd(UNSEEN_1$tprate_transformed, type = 'GEV', method = 'MLE', use.phi = TRUE)
gev_fit_2 <- fevd(UNSEEN_2$tprate_transformed, type = 'GEV', method = 'MLE', use.phi = TRUE)

gumbel_fit_1 <- fevd(UNSEEN_1$tprate_transformed, type = 'Gumbel', method = 'MLE', use.phi = TRUE)
gumbel_fit_2 <- fevd(UNSEEN_2$tprate_transformed, type = 'Gumbel', method = 'MLE', use.phi = TRUE)

return_periods <- c(5, 10, 50, 100, 500)
return_levels_1 <- return.level(gev_fit_1, return.period = return_periods)
return_levels_2 <- return.level(gev_fit_2, return.period = return_periods)

original_scale_return_levels_1 <- 1 / return_levels_1
original_scale_return_levels_2 <- 1 / return_levels_2
print(original_scale_return_levels_1)
print(original_scale_return_levels_2)

# 2012 precipitation was .00126 
