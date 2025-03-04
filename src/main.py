from python.distribution_analysis import DistributionAnalysis
import pandas as pd
import scipy.stats as stats

# Read data from CSV files
control_data = pd.read_csv('src/data/control_group_data.csv')
treatment_data = pd.read_csv('src/data/treatment_group_data.csv')

# Add columns for binary random variables. 1 if insurance was purchased with the booking, 0 otherwise
control_data['insurance_attach_binary'] = control_data['insurance_net_gp_usd'].apply(lambda x: 1 if x > 0 else 0)
treatment_data['insurance_attach_binary'] = treatment_data['insurance_net_gp_usd'].apply(lambda x: 1 if x > 0 else 0)


# Create an instance of DistributionAnalysis
object = DistributionAnalysis(control_data, treatment_data)

# Get binomial statistics for control and treatment groups
n_control, control_proportion, expectation_control, control_stdev_proportion = object.get_binomial_stats(control_data, 'insurance_attach_binary')
n_test, treatment_proportion, expectation_treatment, treatment_stdev_proportion = object.get_binomial_stats(treatment_data, 'insurance_attach_binary')

# Print the proportions, expectations, and standard deviations for both groups
print(f"Treatment Proportion: {treatment_proportion}, Expectation: {expectation_treatment}, Std Dev: {treatment_stdev_proportion}")
print(f"Control Proportion: {control_proportion}, Expectation: {expectation_control}, Std Dev: {control_stdev_proportion}")

# Calculate the difference in means
difference_in_means = treatment_proportion - control_proportion

# Calculate the z-statistic for the difference in proportions
z = difference_in_means / ((treatment_stdev_proportion * n_control + control_stdev_proportion * n_test) / (n_test + n_control))

# Calculate the p-value for the z-statistic
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"P-value: {p_value}")

# Test the goodness of fit for the lognormal distribution
#object.normal_goodness_of_fit('insurance_net_gp_usd')

# Test the goodness of fit for the lognormal distribution
#object.lognormal_goodness_of_fit('insurance_net_gp_usd')

# Compare the means of two columns assuming lognormal distribution
#object.compare_means_lognormal('insurance_net_gp_usd', 'insurance_net_gp_usd')

# Compare the means of two columns assuming normal distribution
#object.compare_means_normal('insurance_net_gp_usd', 'insurance_net_gp_usd')

#plot the histograms for the control and treatment groups

control_no_outliers = object.control_attached['insurance_net_gp_usd'].copy()
control_no_outliers = control_no_outliers[control_no_outliers < object.control_attached['insurance_net_gp_usd'].mean() + object.control_attached['insurance_net_gp_usd'].std()]

treatment_no_outliers = object.treatment_attached['insurance_net_gp_usd'].copy()
treatment_no_outliers = treatment_no_outliers[treatment_no_outliers < object.treatment_attached['insurance_net_gp_usd'].mean() + object.treatment_attached['insurance_net_gp_usd'].std()]

#object.plot_histogram(control_no_outliers, treatment_no_outliers)

#fit a lognormal curve to control_attached['insurance_net_gp_usd'] and treatment_attached['insurance_net_gp_usd']

control_params = object.fit_lognormal(object.control_attached['insurance_net_gp_usd'], 'control')
treatment_params = object.fit_lognormal(object.treatment_attached['insurance_net_gp_usd'], 'treatment')

prob_treatment_greater_than_control = object.probability_treatment_greater_than_control(control_params, treatment_params)
print(f"Probability that treatment group is greater than control group: {prob_treatment_greater_than_control}")