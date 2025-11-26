require(UNSEEN)
require(ggplot2)
require(extRemes)
library(tidyverse)
source('../../../UNSEEN-open-Deep-Sky/src/evt_plot.r')


SEAS5_events <- read.csv("../../data/UNSEEN/heatwave-2012/SEAS5_events.csv")
ERA5_events <- read.csv("../../data/UNSEEN/heatwave-2012/ERA5_events.csv")
SEAS5_events$year <- ymd(paste0(SEAS5_events$year, "-01-01"))
ERA5_events$year <- ymd(paste0(ERA5_events$year, "-01-01"))

UNSEEN_1 <- SEAS5_events[SEAS5_events$number < 25 & 
                           SEAS5_events$year < '1985-02-01' &
                           SEAS5_events$year > '1982-02-01',]

UNSEEN_2 <- SEAS5_events[SEAS5_events$number < 25 & 
                           SEAS5_events$year > '2020-01-01',]

obs_1 <- ERA5_events[ERA5_events$year < '1985-02-01' &
                           ERA5_events$year > '1982-02-01',]

obs_2 <- ERA5_events[ERA5_events$year > '2020-01-01',]

gev_fit_1 <- fevd(UNSEEN_1$t2m, type = 'GEV', use.phi = FALSE)
gev_fit_2 <- fevd(UNSEEN_2$t2m, type = 'GEV', use.phi = FALSE)

gumbel_fit_1 <- fevd(UNSEEN_1$t2m, type = 'Gumbel', use.phi = FALSE)
gumbel_fit_2 <- fevd(UNSEEN_2$t2m, type = 'Gumbel', use.phi = FALSE)

# rvs_1 <- ci.fevd(gumbel_fit_1, alpha = 0.05, type = "return.level", return.period = 60, method = "normal")
# rvs_2 <- ci.fevd(gumbel_fit_2, alpha = 0.05, type = "return.level", return.period = 6, method = "normal")

return_periods <- c(5, 10, 50, 100)
return_levels_1 <- return.level(gumbel_fit_1, return.period = return_periods)
return_levels_2 <- return.level(gumbel_fit_2, return.period = return_periods)

print(return_levels_1)
print(return_levels_2)

# 2012 temperature was 296.061 so first time period would take ~60 years to hit that, whereas today it would take 6. 
plot_return_periods <- c(seq(from = 1.01, to = 1.5, by = 0.1), 1.7, 2, 3, 5, 10, 20, 50, 80, 100, 120, 200, 250, 300, 500, 800, 2000, 5000)
plot_return_levels_1 <- return.level(gumbel_fit_1, return.period = plot_return_periods)
plot_return_levels_2 <- return.level(gumbel_fit_2, return.period = plot_return_periods)

data <- data.frame(plot_rperiods, plot_return_levels_1, plot_return_levels_2)
names(data) <- c('return_period', 'model_1', 'model_2')
write.csv(data, "../Data/drought/v3/temp_return_periods_for_plotting.csv")

