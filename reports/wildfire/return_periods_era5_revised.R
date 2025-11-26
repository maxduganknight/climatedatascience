library(ncdf4)
library(extRemes)
library(ggplot2)
library(dplyr)
library(tidyr)
library(knitr)
library(cowplot)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
source('../../../UNSEEN-open-Deep-Sky/src/evt_plot.r')

# Load data
era5_monthly_max <- read.csv('../../data/UNSEEN/wildfire/cems/preprocessed/ERA5_monthly_max.csv')
era5_monthly_max$time <- as.Date(era5_monthly_max$time)
era5_monthly_max <- era5_monthly_max[order(era5_monthly_max$time), ]

era5_yearly_max <- read.csv('../../data/UNSEEN/wildfire/cems/preprocessed/ERA5_yearly_max.csv')
era5_yearly_max$time <- as.Date(era5_yearly_max$time)
era5_yearly_max$year_index <- year(era5_yearly_max$time)
era5_yearly_max <- era5_yearly_max[order(era5_yearly_max$year_index), ]

era5_p95_count <- read.csv('../../data/UNSEEN/wildfire/cems/preprocessed/ERA5_extreme_fwi_days_N_America.csv')
era5_p95_count$Year <- as.numeric(era5_p95_count$Year)
era5_p95_count <- era5_p95_count[order(era5_p95_count$Year), ]


severity_plot <- ggplot(era5_yearly_max, aes(x = year_index, y = fwinx)) +
  geom_line() +  # Line plot
  labs(title = "Extreme FWI Severity",
       x = "Time",
       y = "Extreme FWI Severity") +
  theme_minimal()  # Optional: Use a minimal theme for a cleaner look


frequency_plot <- ggplot(era5_p95_count, aes(x = Year, y = p95_count)) +
  geom_line() +  # Line plot
  labs(title = "Extreme FWI Frequency",
       x = "Time",
       y = "Extreme FWI Frequency") +
  theme_minimal()  # Optional: Use a minimal theme for a cleaner look

combined_plot <- plot_grid(severity_plot, frequency_plot, labels = "AUTO")
combined_plot


#ggsave(combined_plot, height = 5, width = 7,   filename = "../../figures/UNSEEN_wildfire/extreme_fwi_combined.png")

severity_model <- fevd(x = era5_yearly_max$fwinx, 
                             location.fun = ~ era5_yearly_max$year_index + I(era5_yearly_max$year_index^2), 
                             #scale.fun = ~ era5_yearly_max$index + I(era5_yearly_max$index^2), 
                             use.phi = TRUE, 
                             type = "GEV")

# severity metrics

log_likelihood_severity <- severity_model$results$value
aic_severity <- summary(severity_model)$AIC
bic_severity <- summary(severity_model)$BIC

era5_p95_count$Year_centered <- scale(era5_p95_count$Year, center = TRUE, scale = FALSE)


# frequency metrics
gev_fit_nonstat_p95_acc <- fevd(x = era5_p95_count$p95_count, 
                                location.fun = ~ era5_p95_count$Year_centered + I(era5_p95_count$Year_centered^2), 
                                #scale.fun = ~c(1:length(era5_p95_count$Year)), 
                                use.phi = TRUE, 
                                type = "GEV")

# Extract metrics for the frequency model
log_likelihood_frequency <- gev_fit_nonstat_p95_acc$results$value
aic_frequency <- summary(gev_fit_nonstat_p95_acc)$AIC
bic_frequency <- summary(gev_fit_nonstat_p95_acc)$BIC

# Create a data frame for the metrics
metrics_df <- data.frame(
  Metric = c("Negative Log-Likelihood", "AIC", "BIC"),
  `Severity Model Value` = c(log_likelihood_severity, aic_severity, bic_severity),
  `Frequency Model Value` = c(log_likelihood_frequency, aic_frequency, bic_frequency)
)

# Print the data frame using knitr for a report
kable(metrics_df, caption = "GEV Model Performance Metrics")



### frequency model

# selected one


severity_model <- fevd(x = era5_yearly_max$fwinx, 
                       location.fun = ~ era5_yearly_max$year_index + I(era5_yearly_max$year_index^2), 
                       #scale.fun = ~ era5_yearly_max$index + I(era5_yearly_max$index^2), 
                       use.phi = TRUE, 
                       type = "GEV")

frequency_model <- fevd(x = era5_p95_count$p95_count, 
                        location.fun = ~ era5_p95_count$Year_centered + I(era5_p95_count$Year_centered^2),
                        #scale.fun = ~c(1:length(era5_p95_count$Year)), 
                            use.phi = TRUE, 
                            type = "GEV")


GEV_type <- 'GEV'

# Define a function to calculate return values and confidence intervals
RV_ci <- function(fit, qcov_start, qcov_end, return_periods) {
  ci_start <- ci(fit, alpha = 0.05, type = 'return.level', return.period = return_periods, qcov = qcov_start)
  ci_end <- ci(fit, alpha = 0.05, type = 'return.level', return.period = return_periods, qcov = qcov_end)
  
  results <- data.frame(
    Return_Period = return_periods,
    Start_Central = ci_start[,2],
    Start_Lower = ci_start[, 1],
    Start_Upper = ci_start[, 3],
    End_Central = ci_end[,2],
    End_Lower = ci_end[, 1],
    End_Upper = ci_end[, 3]
  )
  
  return(results)
}

return_periods <- c(2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100)

# severity return levels
severity_min_index <- min(era5_yearly_max$year_index)
severity_max_index <- max(era5_yearly_max$year_index)

severity_qcov_start <- make.qcov(severity_model, vals = list(mu1 = severity_min_index, mu2 = severity_min_index^2))
severity_qcov_end <- make.qcov(severity_model, vals = list(mu1 = severity_max_index, mu2 = severity_max_index^2))

# frequency return levels
frequency_min_index <- min(era5_p95_count$Year)
frequency_max_index <- max(era5_p95_count$Year)

frequency_qcov_start <- make.qcov(frequency_model, vals = list(mu1 = 2013, mu2 = 2013^2))
frequency_qcov_end <- make.qcov(frequency_model, vals = list(mu1 = frequency_max_index, mu2 = frequency_max_index^2))

severity_results <- RV_ci(severity_model, severity_qcov_start, severity_qcov_end, return_periods)
frequency_results <- RV_ci(frequency_model, frequency_qcov_start, frequency_qcov_end, return_periods)


# Create a data frame for severity results
severity_results_combined <- data.frame(
  Return_Period = severity_results$Return_Period,
  Start_Central = severity_results$Start_Central,
  Start_Lower = severity_results$Start_Lower,
  Start_Upper = severity_results$Start_Upper,
  End_Central = severity_results$End_Central,
  End_Lower = severity_results$End_Lower,
  End_Upper = severity_results$End_Upper
)

# Rename columns for better readability
colnames(severity_results_combined) <- c("Return Period", 
                                         "Start Central Estimate", "Start 95% Lower CI", "Start 95% Upper CI",
                                         "End Central Estimate", "End 95% Lower CI", "End 95% Upper CI")

# Create a data frame for frequency results
frequency_results_combined <- data.frame(
  Return_Period = frequency_results$Return_Period,
  Start_Central = frequency_results$Start_Central,
  Start_Lower = frequency_results$Start_Lower,
  Start_Upper = frequency_results$Start_Upper,
  End_Central = frequency_results$End_Central,
  End_Lower = frequency_results$End_Lower,
  End_Upper = frequency_results$End_Upper
)

# Rename columns for better readability
colnames(frequency_results_combined) <- c("Return Period", 
                                          "Start Central Estimate", "Start 95% Lower CI", "Start 95% Upper CI",
                                          "End Central Estimate", "End 95% Lower CI", "End 95% Upper CI")

# Use kable to display the severity table
kable(severity_results_combined, caption = "Return Levels and 95% Confidence Intervals for Severity Model")

# Use kable to display the frequency table
kable(frequency_results_combined, caption = "Return Levels and 95% Confidence Intervals for Frequency Model")

# confidence intervals
plot_severity <- ggplot(severity_results, aes(x = Return_Period)) +
  geom_line(aes(y = Start_Central), col = 'black') +
  geom_ribbon(aes(ymin = Start_Lower, ymax = Start_Upper), fill = 'black', alpha = 0.5) +
  geom_line(aes(y = End_Central), col = 'red') +
  geom_ribbon(aes(ymin = End_Lower, ymax = End_Upper), fill = 'red', alpha = 0.5) +
  scale_x_continuous(trans = 'log10') +
  theme_classic() +
  xlab('Return period (years)') +
  ylab('Severity Return Level')

plot_frequency <- ggplot(frequency_results, aes(x = Return_Period)) +
  geom_line(aes(y = Start_Central), col = 'black') +
  geom_ribbon(aes(ymin = Start_Lower, ymax = Start_Upper), fill = 'black', alpha = 0.5) +
  geom_line(aes(y = End_Central), col = 'red') +
  geom_ribbon(aes(ymin = End_Lower, ymax = End_Upper), fill = 'red', alpha = 0.5) +
  scale_x_continuous(trans = 'log10') +
  theme_classic() +
  xlab('Return period (years)') +
  ylab('Frequency Return Level')

# Combine the plots
combined_plot <- ggarrange(plot_severity, plot_frequency, 
                           labels = c("Severity", "Frequency"),
                           common.legend = TRUE,
                           legend = 'top', 
                           ncol = 2, nrow = 1)

print(combined_plot)
#ggsave(combined_plot, height = 5, width = 7,   filename = "../../figures/UNSEEN_wildfire/combined_return_value_plot.png")

