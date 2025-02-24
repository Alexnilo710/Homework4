import math
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def ln_pdf(D, mu, sigma):
    """Standard log-normal probability density function."""
    if D <= 0:
        return 0
    return (1 / (D * sigma * math.sqrt(2 * math.pi))) * math.exp(-((math.log(D) - mu) ** 2) / (2 * sigma ** 2))


def ln_cdf(D, mu, sigma):
    """Computes the cumulative distribution function for the log-normal distribution using quad."""
    result, _ = quad(lambda x: ln_pdf(x, mu, sigma), 0, D)
    return result


def truncated_pdf(D, mu, sigma, D_Min, D_Max, F_DMin, F_DMax):
    """
    Computes the truncated log-normal PDF:
      f_trunc(D) = ln_pdf(D, mu, sigma) / (F_ln(D_Max) - F_ln(D_Min))
    """
    if D < D_Min or D > D_Max:
        return 0
    return ln_pdf(D, mu, sigma) / (F_DMax - F_DMin)


def truncated_cdf(D, mu, sigma, D_Min, D_Max, F_DMin, F_DMax):
    """
    Computes the truncated cumulative distribution function:
      F_trunc(D) = (ln_cdf(D, mu, sigma) - F_DMin) / (F_DMax - F_DMin)
    """
    if D < D_Min:
        return 0
    if D > D_Max:
        return 1
    return (ln_cdf(D, mu, sigma) - F_DMin) / (F_DMax - F_DMin)


def main():
    # Default parameters
    default_mu = math.log(2)  # Default mean of ln(D)
    default_sigma = 1.0
    default_D_Min = 3.0 / 8.0  # 0.375
    default_D_Max = 1.0

    # Solicit user input for pre-sieved and sieved parameters.
    try:
        mu_in = input(f"Enter the mean of ln(D) [default {default_mu:.3f}]: ").strip()
        mu = float(mu_in) if mu_in else default_mu
        sigma_in = input(f"Enter the standard deviation of ln(D) [default {default_sigma:.3f}]: ").strip()
        sigma = float(sigma_in) if sigma_in else default_sigma
        D_Min_in = input(f"Enter the minimum diameter D_Min [default {default_D_Min:.3f}]: ").strip()
        D_Min = float(D_Min_in) if D_Min_in else default_D_Min
        D_Max_in = input(f"Enter the maximum diameter D_Max [default {default_D_Max:.3f}]: ").strip()
        D_Max = float(D_Max_in) if D_Max_in else default_D_Max
    except Exception as e:
        print("Invalid input. Using default values.")
        mu, sigma, D_Min, D_Max = default_mu, default_sigma, default_D_Min, default_D_Max

    # Compute the untruncated CDF at D_Min and D_Max for normalization.
    F_DMin = ln_cdf(D_Min, mu, sigma)
    F_DMax = ln_cdf(D_Max, mu, sigma)

    # Create a numpy array for D values from D_Min to D_Max.
    D_vals = np.linspace(D_Min, D_Max, 500)

    # Compute truncated PDF and CDF values using vectorized functions.
    vec_trunc_pdf = np.vectorize(lambda D: truncated_pdf(D, mu, sigma, D_Min, D_Max, F_DMin, F_DMax))
    vec_trunc_cdf = np.vectorize(lambda D: truncated_cdf(D, mu, sigma, D_Min, D_Max, F_DMin, F_DMax))

    pdf_vals = vec_trunc_pdf(D_vals)
    cdf_vals = vec_trunc_cdf(D_vals)

    # Define the upper limit for integration:
    D_upper = D_Min + 0.75 * (D_Max - D_Min)
    F_D_upper = truncated_cdf(D_upper, mu, sigma, D_Min, D_Max, F_DMin, F_DMax)

    # Create the plots.
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

    # Upper subplot: truncated PDF
    ax1.plot(D_vals, pdf_vals, 'b-', label='Truncated log-normal PDF')
    ax1.set_ylabel('f(D)', fontsize=12)
    ax1.set_title('Truncated Log-Normal PDF and CDF', fontsize=14)
    ax1.set_xlim(D_Min, D_Max)
    ax1.set_ylim(0, max(pdf_vals) * 1.1)

    # Fill the area under the PDF from D_Min to D_upper.
    D_fill = np.linspace(D_Min, D_upper, 200)
    pdf_fill = vec_trunc_pdf(D_fill)
    ax1.fill_between(D_fill, pdf_fill, color='grey', alpha=0.3)

    # Annotate the upper plot with F(D_upper)
    text_x = D_Min + 0.1 * (D_Max - D_Min)
    text_y = max(pdf_vals) * 0.7
    ax1.text(text_x, text_y, f'$F(D)={F_D_upper:0.3f}$ at $D={D_upper:0.3f}$', fontsize=10)
    ax1.axvline(D_upper, color='black', linestyle='--', linewidth=1)
    ax1.axhline(0, color='black', linewidth=1)

    # Lower subplot: truncated CDF
    ax2.plot(D_vals, cdf_vals, 'r-', label='Truncated log-normal CDF')
    ax2.set_ylabel('F(D)', fontsize=12)
    ax2.set_xlabel('D', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.plot(D_upper, F_D_upper, 'ko')  # mark the point at D_upper
    ax2.axvline(D_upper, color='black', linestyle='--', linewidth=1)
    ax2.axhline(F_D_upper, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()