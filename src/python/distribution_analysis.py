import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math


class DistributionAnalysis():
    def __init__(self, control_data, treatment_data):
        self.control_data = control_data
        self.treatment_data = treatment_data
        self.control_attached = self.control_data[self.control_data['insurance_net_gp_usd'] > 0]
        self.treatment_attached = self.treatment_data[self.treatment_data['insurance_net_gp_usd'] > 0]

    def get_binomial_stats(self, df, binomial_column):
        n = len(df)
        p = sum(df[binomial_column])/n
        expectation = n * p
        stdev = (n * p * (1 - p)) ** 0.5
        return n, p, expectation, stdev

    def is_good_fit_for_normal_approximation(self, n, p):
        if n * p >= 10 and n * (1 - p) >= 10:
            return True, "np = " + str(n * p) + " and n(1-p) = " + str(n * (1 - p))
        else:
            return False, "np = " + str(n * p) + " and n(1-p) = " + str(n * (1 - p))

    def normal_approx_to_binomial_dist(self, n, p):
        output_vector = n * [0]
        for i in range(0, n):
            NchooseK = math.comb(n, i)
            output_vector[i] = NchooseK * (p ** i) * ((1 - p) ** (n - i))
        return output_vector

    def normal_dist(self, mu, sigmaSquare, n):
        normal_vector = n * [0]
        for i in range(0, n):
            normal_vector[i] = 1 / (2 * math.pi * sigmaSquare) ** 0.5 * math.exp(-((i - mu) ** 2) / (2 * sigmaSquare))
        return normal_vector

    def test_normality(self, control_column, test_column):
        # Perform a Shapiro-Wilk test for normality
        control_shapiro = stats.shapiro(self.control_data[control_column])
        test_shapiro = stats.shapiro(self.treatment_data[test_column])

        print(f"Shapiro-Wilk test for normality of control group: {control_shapiro}")
        print(f"Shapiro-Wilk test for normality of treatment group: {test_shapiro}")

    def binary_test(self, control_proportion,test_proportion,control_stdev_proportion,test_stdev_proportion):
        z = (test_proportion - control_proportion) / (control_stdev_proportion ** 2 + test_stdev_proportion ** 2) ** 0.5
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        print(f"Control group fraction: {control_proportion}")
        print(f"Treatment group fraction: {test_proportion}")
        print(f"Z statistic: {z}")
        print(f"P-value: {p_value}")

    def normal_goodness_of_fit(self, data_column):
        # Fit a normal distribution to the data
        mu, std = stats.norm.fit(self.control_attached[data_column])

        # Plot the Q-Q plot for the normal distribution
        plt.figure()
        stats.probplot(self.control_attached[data_column], dist="norm", sparams=(mu, std), plot=plt)
        plt.title("Q-Q Plot for Normal Distribution")
        plt.show()

        # Perform the Kolmogorov-Smirnov test
        d, p_value = stats.kstest(self.control_attached[data_column], 'norm', args=(mu, std))
        print(f"Kolmogorov-Smirnov test statistic: {d}")
        print(f"P-value: {p_value}")

    def lognormal_goodness_of_fit(self, data_column):
        # Fit a lognormal distribution to the data
        shape, loc, scale = stats.lognorm.fit(self.control_attached[data_column])

        # Plot the Q-Q plot for the lognormal distribution
        plt.figure()
        stats.probplot(self.control_attached[data_column], dist="lognorm", sparams=(shape, loc, scale), plot=plt)
        plt.title("Q-Q Plot for Lognormal Distribution")
        plt.show()

        # Perform the Kolmogorov-Smirnov test
        d, p_value = stats.kstest(self.control_attached[data_column], 'lognorm', args=(shape, loc, scale))
        print(f"Kolmogorov-Smirnov test statistic: {d}")
        print(f"P-value: {p_value}")

    def beta_goodness_of_fit(self, data_column):
        # Scale the data to the [0, 1] range
        data = self.control_data[data_column]
        scaled_data = (data - data.min()) / (data.max() - data.min())

        # Fit a beta distribution to the scaled data
        a, b, loc, scale = stats.beta.fit(scaled_data, floc=0, fscale=1)

        # Plot the Q-Q plot for the beta distribution
        plt.figure()
        stats.probplot(scaled_data, dist="beta", sparams=(a, b, loc, scale), plot=plt)
        plt.title("Q-Q Plot for Beta Distribution")
        plt.show()

        # Perform the Kolmogorov-Smirnov test
        d, p_value = stats.kstest(scaled_data, 'beta', args=(a, b, loc, scale))
        print(f"Kolmogorov-Smirnov test statistic: {d}")
        print(f"P-value: {p_value}")

    def compare_means_lognormal(self, column1, column2):
        # Log-transform the data
        log_data1 = np.log(self.control_attached[column1])
        log_data2 = np.log(self.treatment_attached[column2])

        # Calculate means and variances
        mean1, mean2 = np.mean(log_data1), np.mean(log_data2)
        var1, var2 = np.var(log_data1, ddof=1), np.var(log_data2, ddof=1)
        n1, n2 = len(log_data1), len(log_data2)

        # Calculate the z-statistic
        z = (mean1 - mean2) / np.sqrt(var1 / n1 + var2 / n2)

        # Calculate the p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        print(f"Z-statistic: {z}")
        print(f"P-value: {p_value}")

    def compare_means_normal(self, column1, column2):
        # Calculate means and variances
        mean1, mean2 = np.mean(self.control_attached[column1]), np.mean(self.treatment_attached[column2])
        var1, var2 = np.var(self.control_attached[column1], ddof=1), np.var(self.treatment_attached[column2], ddof=1)
        n1, n2 = len(self.control_attached[column1]), len(self.treatment_attached[column2])

        # Calculate the z-statistic
        z = (mean1 - mean2) / np.sqrt(var1 / n1 + var2 / n2)

        # Calculate the p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        print(f"Z-statistic: {z}")
        print(f"P-value: {p_value}")

    def plot_histogram(self, control, treatment):
        plt.figure()
        plt.hist(control, bins=100, alpha=0.5, label='Control Group')
        plt.hist(treatment, bins=100, alpha=0.5, label='Treatment Group')
        plt.legend()
        plt.xlabel('Insurance Net GP USD')
        plt.ylabel('Frequency')
        plt.title('Histogram of Insurance Net GP USD')
        plt.show()

    def fit_lognormal(self, data, label):
        # Fit a lognormal distribution to the data
        shape, loc, scale = stats.lognorm.fit(data, floc=0)
        
        # Generate x values for the fitted curve
        x = np.linspace(data.min(), data.max(), 10000)
        
        # Generate the fitted curve
        fitted_curve = stats.lognorm.pdf(x, shape, loc, scale)
        
        # Plot the histogram and the fitted curve
        #plt.figure()
        #plt.hist(data, bins=40, density=True, alpha=0.6, color='g', label='Histogram')
        #plt.plot(x, fitted_curve, 'r-', lw=2, label='Fitted Lognormal Curve')
        #plt.title(f'Lognormal Fit for {label}')
        #plt.xlabel('Value')
        #plt.ylabel('Density')
        #plt.legend()
        #plt.show()

        return shape, loc, scale
    def probability_treatment_greater_than_control(self, control_params, treatment_params, num_samples=10000):
        # Unpack parameters
        shape_control, loc_control, scale_control = control_params
        shape_treatment, loc_treatment, scale_treatment = treatment_params

        # Sample from the lognormal distributions
        control_samples = stats.lognorm.rvs(shape_control, loc_control, scale_control, size=num_samples)
        treatment_samples = stats.lognorm.rvs(shape_treatment, loc_treatment, scale_treatment, size=num_samples)

        # Calculate the probability that treatment samples are greater than control samples
        prob = np.mean(treatment_samples > control_samples)

        return prob

