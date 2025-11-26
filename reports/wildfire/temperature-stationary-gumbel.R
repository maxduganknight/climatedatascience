require(UNSEEN)
require(ggplot2)
require(extRemes)
library(tidyverse)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

source('../../../UNSEEN-open-Deep-Sky/src/evt_plot.r')

SEAS5_events <- read.csv("../../data/UNSEEN/wildfire/preprocessed/SEAS5_CA_events.csv")
ERA5_events <- read.csv("../../data/UNSEEN/wildfire/preprocessed/ERA5_CA_events.csv")

SEAS5_events$year <- ymd(paste0(SEAS5_events$year, "-01-01"))
ERA5_events$year <- ymd(paste0(ERA5_events$year, "-01-01"))

SEAS5_events <- SEAS5_events[SEAS5_events$number < 25,]

SEAS5_events_hindcast <- SEAS5_events[
  SEAS5_events$year < '2017-02-01',]

ERA5_events_hindcast <- ERA5_events[
  ERA5_events$year > '1981-02-01' &
    ERA5_events$year < '2017-02-01',]


obs <- ERA5_events[
  ERA5_events$year > '1981-02-01',]

UNSEEN_bc <- SEAS5_events[SEAS5_events$number < 25,]

UNSEEN_bc$t2m <- (UNSEEN_bc$t2m +
                    mean(ERA5_events_hindcast$t2m) - mean(SEAS5_events_hindcast$t2m)
)

UNSEEN_bc$d2m <- (UNSEEN_bc$d2m +
                    mean(ERA5_events_hindcast$d2m) - mean(SEAS5_events_hindcast$d2m)
)

UNSEEN_1 <- UNSEEN_bc[UNSEEN_bc$year < '1998-01-01' &
                        UNSEEN_bc$year > '1981-01-01',]

UNSEEN_2 <- UNSEEN_bc[UNSEEN_bc$year >= '2008-01-01',]

obs_1 <- ERA5_events[ERA5_events$year < '1985-02-01' &
                           ERA5_events$year > '1982-02-01',]

obs_2 <- ERA5_events[ERA5_events$year > '2020-01-01',]

########################################################
# plot

timeseries = unseen_timeseries(
  ensemble = UNSEEN_bc,
  obs = obs,
  ensemble_yname = "t2m",
  ensemble_xname = "year",
  obs_yname = "t2m",
  obs_xname = "year",
  ylab = "June, July, August Minnesota temperature (C)")

timeseries + theme(text = element_text(size = 14))

########################################################




gev_fit_1 <- fevd(UNSEEN_1$t2m, type = 'GEV', use.phi = FALSE)
gev_fit_2 <- fevd(UNSEEN_2$t2m, type = 'GEV', use.phi = FALSE)

gumbel_fit_1 <- fevd(UNSEEN_1$t2m, type = 'Gumbel', use.phi = FALSE)
gumbel_fit_2 <- fevd(UNSEEN_2$t2m, type = 'Gumbel', use.phi = FALSE)

# rvs_1 <- ci.fevd(gev_fit_1, alpha = 0.05, type = "return.level", return.period = 100, method = "normal")
# rvs_2 <- ci.fevd(gev_fit_2, alpha = 0.05, type = "return.level", return.period = 5, method = "normal")

return_periods <- c(5, 10, 20, 40, 50, 70, 100, 120)
return_levels_1 <- return.level(gumbel_fit_1, return.period = return_periods)
return_levels_2 <- return.level(gumbel_fit_2, return.period = return_periods)

print(return_levels_1)
print(return_levels_2)

# Calculate the 100-year return level for UNSEEN_1
return_level_100yr_1 <- return.level(gumbel_fit_1, return.period = 100)
return_level_100yr_value <- return_level_100yr_1


# Use uniroot to find the return period in UNSEEN_2 for the 100-year event level of UNSEEN_1
return_period_in_UNSEEN_2 <- function(level) {
  # Function to find the return period where return level matches the level from UNSEEN_1
  fn <- function(period) {
    tryCatch({
      # Extract the return level at the given period
      level_at_period <- return.level(gumbel_fit_2, return.period = period)
      level_at_period - level
    }, error = function(e) {
      # Return a high number to move the root finder away from problematic values
      1e6
    })
  }
  
  # Use uniroot to find the root, starting from just above 1 to avoid the error and assuming a wide range to search within
  result <- uniroot(fn, interval = c(1.01, 1000), extendInt = "yes")
  if (result$f.root == 1e6) {
    return(NA)  # Return NA if no valid root found within the range
  } else {
    return(result$root)
  }
}

# Calculate the return period in UNSEEN_2 for the 100-year event level from UNSEEN_1
calculated_return_period <- return_period_in_UNSEEN_2(return_level_100yr_value)
if (is.na(calculated_return_period)) {
  print("The return period could not be calculated within the given range.")
} else {
  print(paste("The return period in UNSEEN_2 for the 100-year level event observed in UNSEEN_1 is approximately:", calculated_return_period, "years."))
}



