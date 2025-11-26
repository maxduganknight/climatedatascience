library(ncdf4)
library(extRemes)
library(ggplot2)
library(dplyr)
library(tidyr)
library(knitr)
library(cowplot)
library(lubridate)
library(POT)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

era5 <- read.csv('../../../data/UNSEEN/hurricane/rainfall_preprocessed/ERA5_1d_monthly_max.csv')
era5$time <- ymd(era5$time)
era5$month <- month(era5$time)
era5$year <- year(era5$time)

era5 <- era5[!is.na(era5$tp), ]

# visualize original data
ggplot(era5, aes(x = year, y = tp)) +
  geom_point() +
  geom_smooth(method = "loess") +
  ggtitle("Single-day Monthly Maximum Precipitation (m)")
#ggsave(filename = '../../../figures/UNSEEN_hurricane/rainfall/technical_appendix/original_precipitation_data.png')

stationary_model <- fevd(era5$tp, type = "GEV")

non_stationary_model <- fevd(era5$tp, 
                                scale.fun = ~era5$year, 
                                location.fun = ~ era5$year, 
                                #use.phi = TRUE, 
                                type = "GEV")

lr.test(stationary_model, non_stationary_model)

## return periods

GEV_type <- "GEV"

# Define a function to calculate return values and confidence intervals
RV_ci <- function(fit, qcov_start, qcov_end, return_periods) {
  ci_start <- ci(fit, alpha = 0.05, type = 'return.level', return.period = return_periods, qcov = qcov_start, method = 'normal')
  ci_end <- ci(fit, alpha = 0.05, type = 'return.level', return.period = return_periods, qcov = qcov_end, method = 'normal')
  
  results <- data.frame(
    Return_Period = return_periods,
    Start_Lower = ci_start[, 1],
    Start_Central = ci_start[, 2],
    Start_Upper = ci_start[, 3],
    End_Lower = ci_end[, 1],
    End_Central = ci_end[, 2],
    End_Upper = ci_end[, 3]
  )
  
  return(results)
}

return_periods <- c(2, 3, 5, 10, 25, 100)

# Extract earliest and latest years
earliest_year <- 1983
latest_year <- 2023

# Create covariate matrices for the earliest and latest years
# just location
# qcov_earliest <- make.qcov(non_stationary_model, vals = list(sigma1 = earliest_year))
# qcov_latest <- make.qcov(non_stationary_model, vals = list(sigma1 = latest_year))

# location and scale
qcov_earliest <- make.qcov(non_stationary_model, vals = list(mu1 = 2018, sigma1 = 2018))
qcov_latest <- make.qcov(non_stationary_model, vals = list(mu1 = latest_year, sigma1 = latest_year))


return_levels_results <- RV_ci(non_stationary_model, qcov_earliest, qcov_latest, return_periods)
print(return_levels_results)

return_levels_results$End_Central[length(return_levels_results$End_Central)] / return_levels_results$Start_Central[length(return_levels_results$Start_Central)]

return_levels_results$End_Central[4] - return_levels_results$Start_Central[4] / return_levels_results$Start_Central[4] * 100
print(increase_pct)

# For technical appendix

log_likelihood <- non_stationary_model$results$value
aic <- summary(non_stationary_model)$AIC
bic <- summary(non_stationary_model)$BIC

metrics_df <- data.frame(
  Metric = c("Negative Log-Likelihood", "AIC", "BIC"),
  `Severity Model Value` = c(log_likelihood, aic, bic)
)

ggplot(return_levels_results, aes(x = Return_Period)) +
  geom_line(aes(y = Start_Central), color = 'blue') +
  geom_ribbon(aes(ymin = Start_Lower, ymax = Start_Upper), alpha = 0.2, fill = 'blue') +
  geom_line(aes(y = End_Central), color = 'red') +
  geom_ribbon(aes(ymin = End_Lower, ymax = End_Upper), alpha = 0.2, fill = 'red') +
  labs(title = "Return Levels Over Time",
       x = "Return Period",
       y = "Return Level") +
  theme_bw()

ggsave(filename = '../../../figures/UNSEEN_hurricane/rainfall/technical_appendix/return_levels_plot.png')

plot(non_stationary_model)


######################################################################
# frequency
######################################################################

# MDK I'm actually not sure this is strictly necessary. The results line up with the results from the first model. But maybe good for confirmation.

tcplot(era5$tp, u.range = c(0.9, 0.999))
mrlplot(era5$tp, nint = 10, alpha = 0.05)

coefficients <- non_stationary_model$results$par
mu0 <- coefficients["mu0"]
mu1 <- coefficients["mu1"]
sigma0 <- coefficients["sigma0"]
sigma1 <- coefficients["sigma1"]
shape <- coefficients["shape"]

location_start <- mu0 + mu1 * start_year
location_end <- mu0 + mu1 * latest_year
scale_start <- sigma0 + sigma1 * start_year
scale_end <- sigma0 + sigma1 * latest_year

calculate_exceedance_probability <- function(location, scale, shape, threshold) {
  if (shape == 0) {
    exceedance_prob <- exp(-(threshold - location) / scale)
  } else {
    exceedance_prob <- 1 - exp(-(1 + shape * (threshold - location) / scale)^(-1 / shape))
  }
  return(exceedance_prob)
}

# Define the threshold for exceedance
quantile(era5$tp, 0.999)
threshold <- 5.639

# Calculate exceedance probabilities for start and end years
prob_exceed_start <- calculate_exceedance_probability(location_start, scale_start, shape, threshold)
prob_exceed_end <- calculate_exceedance_probability(location_end, scale_end, shape, threshold)

cat("Probability of exceeding the threshold at start year (", start_year, "): ", prob_exceed_start, "\n")
cat("Probability of exceeding the threshold at end year (", latest_year, "): ", prob_exceed_end, "\n")

# Calculate the percentage increase in exceedance probability
percentage_increase <- ((prob_exceed_end - prob_exceed_start) / prob_exceed_start) * 100

cat("Extreme precipitation events above the threshold have become ", percentage_increase, "% more likely over the course of the full time range in the data.\n")


### visualize

years <- seq(start_year, latest_year)

# Calculate location and scale parameters for each year
locations <- mu0 + mu1 * years
scales <- sigma0 + sigma1 * years

# Calculate exceedance probabilities for each year
exceedance_probs <- sapply(1:length(years), function(i) {
  calculate_exceedance_probability(locations[i], scales[i], shape, threshold)
})

# Create a data frame for plotting
data <- data.frame(Year = years, Exceedance_Probability = exceedance_probs)

# Plot the exceedance probabilities over time
ggplot(data, aes(x = Year, y = Exceedance_Probability)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 2) +
  labs(title = "Change in Exceedance Probability Over Time",
       x = "Year",
       y = "Exceedance Probability") +
  theme_minimal()

#ggsave(filename = '../../../figures/UNSEEN_hurricane/rainfall/technical_appendix/exceedance_probability.png')

### make some predictions into the future
params <- non_stationary_model$results$par
mu0 <- params[1]
mu1 <- params[2]
sigma0 <- params[3]
sigma1 <- params[4]
shape <- params[5]

# Specify the year for prediction
year <- 2026

# Calculate the location and scale parameters for the given year
mu <- mu0 + mu1 * year
sigma <- sigma0 + sigma1 * year

# Generate random samples from the GEV distribution
library(evd)
set.seed(123) 
predicted_values <- rgev(1000, loc = mu, scale = sigma, shape = shape)

# Summary of predicted values
summary(predicted_values)