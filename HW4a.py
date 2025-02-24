# Imports
import math
from random import random as rnd
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve


def ln_PDF(args):
    """
    Computes the log-normal probability density function.
    :param args: (D, mu, sigma)
    :return: f(D)
    """
    D, mu, sig = args
    if D == 0.0:
        return 0.0
    p = 1 / (D * sig * math.sqrt(2 * math.pi))
    exponent = -((math.log(D) - mu) ** 2) / (2 * sig ** 2)
    return p * math.exp(exponent)


def tln_PDF(args):
    """
    Computes the truncated log-normal probability density function.
    :param args: (D, mu, sig, F_DMin, F_DMax)
    :return: f_trunc(D)
    """
    D, mu, sig, F_DMin, F_DMax = args
    return ln_PDF((D, mu, sig)) / (F_DMax - F_DMin)


def F_tlnpdf(args):
    """
    Integrates the truncated log-normal PDF from D_Min to D.
    :param args: (mu, sig, D_Min, D_Max, D, F_DMax, F_DMin)
    :return: The cumulative probability from D_Min up to D.
    """
    mu, sig, D_Min, D_Max, D, F_DMax, F_DMin = args
    # Ensure D is within the truncated interval.
    if D > D_Max or D < D_Min:
        return 0
    P, err = quad(lambda x: tln_PDF((x, mu, sig, F_DMin, F_DMax)), D_Min, D)
    return P


def makeSample(args, N=100):
    """
    Generates a sample of rock sizes by inverting the truncated CDF.
    For each of the N uniformly random probabilities, fsolve finds the D such that:
        F_tlnpdf((mu, sig, D_Min, D_Max, D, F_DMax, F_DMin)) - P = 0.
    :param args: (ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin)
    :param N: Sample size.
    :return: List of rock diameters.
    """
    ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin = args
    probs = [rnd() for _ in range(N)]
    # Use fsolve to invert the CDF for each random probability.
    d_s = [fsolve(lambda D: F_tlnpdf((ln_Mean, ln_sig, D_Min, D_Max, D, F_DMax, F_DMin)) - P,
                  (D_Max + D_Min) / 2)[0] for P in probs]
    return d_s


def sampleStats(D, doPrint=False):
    """
    Computes the sample mean and variance.
    :param D: List of values.
    :param doPrint: If True, prints the stats.
    :return: (mean, variance)
    """
    N = len(D)
    mean = sum(D) / N
    var = sum((d - mean) ** 2 for d in D) / (N - 1)
    if doPrint:
        print(f"mean = {mean:0.3f}, var = {var:0.3f}")
    return (mean, var)


def getPreSievedParameters(args):
    """
    Prompts the user for the pre-sieved log-normal parameters.
    :param args: (default_mean_ln, default_sig_ln)
    :return: (mean_ln, sig_ln)
    """
    mean_ln, sig_ln = args
    st_mean = input(
        f'Mean of ln(D) for pre-sieved rocks? (default ln({math.exp(mean_ln):0.1f})={mean_ln:0.3f}): ').strip()
    mean_ln = mean_ln if st_mean == '' else float(st_mean)
    st_sig = input(f'Standard deviation of ln(D) for pre-sieved rocks? (default {sig_ln:0.3f}): ').strip()
    sig_ln = sig_ln if st_sig == '' else float(st_sig)
    return (mean_ln, sig_ln)


def getSieveParameters(args):
    """
    Prompts the user for the sieve apertures.
    :param args: (default_D_Min, default_D_Max)
    :return: (D_Min, D_Max)
    """
    D_Min, D_Max = args
    st_D_Max = input(f'Large aperture size? (default {D_Max:0.3f}): ').strip()
    D_Max = D_Max if st_D_Max == '' else float(st_D_Max)
    st_D_Min = input(f'Small aperture size? (default {D_Min:0.3f}): ').strip()
    D_Min = D_Min if st_D_Min == '' else float(st_D_Min)
    return (D_Min, D_Max)


def getSampleParameters(args):
    """
    Prompts the user for sample configuration.
    :param args: (default_N_samples, default_N_sampleSize)
    :return: (N_samples, N_sampleSize)
    """
    N_samples, N_sampleSize = args
    st_samples = input(f'How many samples? (default {N_samples}): ').strip()
    N_samples = N_samples if st_samples == '' else int(st_samples)
    st_size = input(f'How many items per sample? (default {N_sampleSize}): ').strip()
    N_sampleSize = N_sampleSize if st_size == '' else int(st_size)
    return (N_samples, N_sampleSize)


def getFDMaxFDMin(args):
    """
    Computes the cumulative probabilities F_DMax and F_DMin for the untruncated log-normal distribution.
    :param args: (mean_ln, sig_ln, D_Min, D_Max)
    :return: (F_DMin, F_DMax)
    """
    mean_ln, sig_ln, D_Min, D_Max = args
    F_DMax, _ = quad(lambda x: ln_PDF((x, mean_ln, sig_ln)), 0, D_Max)
    F_DMin, _ = quad(lambda x: ln_PDF((x, mean_ln, sig_ln)), 0, D_Min)
    return (F_DMin, F_DMax)


def makeSamples(args):
    """
    Generates multiple samples and computes their means.
    :param args: (mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize, N_samples, doPrint)
    :return: (Samples, Means)
    """
    mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize, N_samples, doPrint = args
    Samples = []
    Means = []
    for n in range(N_samples):
        sample = makeSample((mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin), N=N_sampleSize)
        Samples.append(sample)
        stats_sample = sampleStats(sample)
        Means.append(stats_sample[0])
        if doPrint:
            print(f"Sample {n}: mean = {stats_sample[0]:0.3f}, var = {stats_sample[1]:0.3f}")
    return Samples, Means


def main():
    """
    Simulates the gravel production process:
      - Requests pre-sieved log-normal parameters and sieve sizes.
      - Computes the truncated log-normal CDF normalization factors.
      - Generates samples from the truncated distribution.
      - Computes and displays sample statistics.
    """
    # Default parameters
    mean_ln = math.log(2)  # ln(mean) with D in inches
    sig_ln = 1.0
    D_Max = 1.0
    D_Min = 3.0 / 8.0
    N_samples = 11
    N_sampleSize = 100
    goAgain = True

    while goAgain:
        mean_ln, sig_ln = getPreSievedParameters((mean_ln, sig_ln))
        D_Min, D_Max = getSieveParameters((D_Min, D_Max))
        N_samples, N_sampleSize = getSampleParameters((N_samples, N_sampleSize))
        F_DMin, F_DMax = getFDMaxFDMin((mean_ln, sig_ln, D_Min, D_Max))

        # Generate samples and compute statistics.
        Samples, Means = makeSamples((mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin,
                                      N_sampleSize, N_samples, True))
        overall_stats = sampleStats(Means)
        print(f"Mean of the sampling means: {overall_stats[0]:0.3f}")
        print(f"Variance of the sampling means: {overall_stats[1]:0.6f}")

        again = input("Go again? (y/n): ").strip().lower()
        goAgain = again.startswith('y')


if __name__ == '__main__':
    main()
