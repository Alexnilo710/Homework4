# region imports
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# endregion

# region functions
def main():
    '''
    Calculates P(x<1|N(0,1)) and P(x>μ+2σ|N(175, 3)) and displays both the GNPDF and CDF
    for each case
    :return: nothing
    '''
    #part 1 P(x<1|N(0,1))
    mu_a = 0.0 #mean
    sig_a = 1.0  #standard deviation
    c_a = 1.0

    p_a = stats.norm(mu_a,sig_a).cdf(c_a)  #calculate the probability P(x<1|N(0,1))

    #create the illustrative plots for part a
    x_a=np.linspace(mu_a-5*sig_a,mu_a+5*sig_a,500) #create a numpy array using linspace between mu-5*sigma to mu+5*sigma with 500 points
    cdf_a = np.array([stats.norm(mu_a,sig_a).cdf(x) for x in x_a]) #create a numpy array filled with values of CDF
    gnpdf_a = np.array([stats.norm(mu_a, sig_a).pdf(x) for x in x_a])  #create a numpy array for f(x) from the GNPDF

    plt.subplots(2,1,sharex=True) #create two, stacked plots using subplots with sharex=True
    plt.subplot(2, 1, 1) #set subplot 1 for our focus by using plt.subplot
    plt.plot(x_a, gnpdf_a)  #plot the gndpf_a vs x_a
    plt.xlim(x_a.min(),x_a.max())
    plt.ylim(0, gnpdf_a.max()*1.1)
    # fill in area below GNPDF in range mu_a-5*sig_a to 1
    x_fill = np.linspace(mu_a - 5 * sig_a, c_a, 100) #create a numpy array of x values from mu-5*sigma to 1 with 100 points
    gnpdf_fill = np.array([stats.norm(mu_a,sig_a).pdf(x) for x in x_fill]) #calculate the GNPDF function for each x in x_fill and store in numpy array
    ax=plt.gca() #get the axes for the current plot
    ax.fill_between(x_fill, gnpdf_fill, color='grey', alpha=0.3) #create the filled region between gnpdf and x axis

    #construct the equation to display on GNPDF using TeX
    text_x=mu_a-4*sig_a
    text_y=0.65*gnpdf_a.max()
    plt.text(text_x,text_y,r'$f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$')
    arrow_x=(c_a-mu_a+5*sig_a)*2/3+(mu_a-5*sig_a) #calculate the x coordinate for where the arrow should point
    arrow_y=(stats.norm(mu_a, sig_a).pdf(arrow_x)/2.0 ) #calculate the y coordinate for where the arrow should point
    plt.annotate('P(x<{:0.2f}|N({:0.2f},{:0.2f})={:0.2f}'.format(c_a, mu_a, sig_a, p_a), size=8,xy=(arrow_x,arrow_y),xytext=(text_x,0.5*text_y),arrowprops=dict(arrowstyle='->', connectionstyle="arc3")) #draw the arrow with text
    plt.ylabel('f(x)', size=12)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=10)  # format tick marks
    # ax.xaxis.set_ticklabels([]) #erase x tick labels for the top graph
    ax.yaxis.set_label('f(x)')

    #create the CDF plot
    plt.subplot(2,1,2) #select the second plot
    plt.plot(x_a,cdf_a) #plot cdf_a vs x_a
    plt.ylim(0,1)
    plt.ylabel('$\Phi(x)=\int_{-\infty}^{x}f(x)\mathrm{d}x$', size=12)
    plt.xlabel('x')
    plt.plot(c_a,p_a,'o', markerfacecolor='white', markeredgecolor='red')
    ax=plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=10)  # format tick marks
    ax.set_xlim(ax.get_xlim())

    ax.hlines(p_a,ax.get_xlim()[0],c_a, color='black', linewidth=1)
    ax.vlines(c_a, 0, p_a,color='black', linewidth=1)
    plt.show()

    #part 2 P(x>mu+2*sigma|N(175,3))
    mu_b = 175.0 #mean
    sig_b = 3.0  #standard deviation
    c_b = mu_b+2*sig_b

    p_b = 1.0-stats.norm(mu_b,sig_b).cdf(c_b)  #calculate the probability P(x<1|N(0,1))

    #create the illustrative plots for part a
    x_b=np.linspace(mu_b-5*sig_b,mu_b+5*sig_b,500) #create a numpy array using linspace between mu-5*sigma to mu+5*sigma with 500 points
    cdf_b = np.array([1.0-stats.norm(mu_b,sig_b).cdf(x) for x in x_b]) #create a numpy array filled with values of CDF
    gnpdf_b = np.array([stats.norm(mu_b, sig_b).pdf(x) for x in x_b])  #create a numpy array for f(x) from the GNPDF

    plt.subplots(2,1,sharex=True) #create two, stacked plots using subplots with sharex=True
    plt.subplot(2, 1, 1) #set subplot 1 for our focus by using plt.subplot
    plt.plot(x_b, gnpdf_b)  #plot the gndpf_b vs x_b
    plt.xlim(x_b.min(),x_b.max())
    plt.ylim(0, gnpdf_b.max()*1.1)
    # fill in area below GNPDF in range mu_b-5*sig_b to 1
    x_fill = np.linspace(c_b,mu_b + 5 * sig_b, 100) #create a numpy array of x values from mu-5*sigma to 1 with 100 points
    gnpdf_fill = np.array([stats.norm(mu_b,sig_b).pdf(x) for x in x_fill]) #calculate the GNPDF function for each x in x_fill and store in numpy array
    ax=plt.gca() #get the axes for the current plot
    ax.fill_between(x_fill, gnpdf_fill, color='grey', alpha=0.3) #create the filled region between gnpdf and x axis

    #construct the equation to display on GNPDF using TeX
    text_x=c_b+sig_b
    text_y=0.65*gnpdf_b.max()
    plt.text(text_x,text_y,r'$f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$')
    arrow_x=(c_b)+sig_b*1/2 #calculate the x coordinate for where the arrow should point
    arrow_y=(stats.norm(mu_b, sig_b).pdf(arrow_x)/2.0 ) #calculate the y coordinate for where the arrow should point
    plt.annotate('P(x>{:0.2f}|N({:0.2f},{:0.2f})={:0.2f}'.format(c_b, mu_b, sig_b, p_b), size=8,xy=(arrow_x,arrow_y),xytext=(text_x,0.5*text_y),arrowprops=dict(arrowstyle='->', connectionstyle="arc3")) #draw the arrow with text
    plt.ylabel('f(x)', size=12)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=10)  # format tick marks
    # ax.xaxis.set_ticklabels([]) #erase x tick labels for the top graph
    ax.yaxis.set_label('f(x)')

    #create the CDF plot
    plt.subplot(2,1,2) #select the second plot
    plt.plot(x_b,cdf_b) #plot cdf_b vs x_b
    plt.ylim(0,1)
    plt.ylabel('$1-\Phi(x)=1-\int_{-\infty}^{x}f(x)\mathrm{d}x$', size=12)
    plt.xlabel('x')
    plt.plot(c_b,p_b,'o', markerfacecolor='white', markeredgecolor='red')
    ax=plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=10)  # format tick marks
    ax.set_xlim(ax.get_xlim())

    ax.hlines(p_b,ax.get_xlim()[0],c_b, color='black', linewidth=1)
    ax.vlines(c_b, 0, p_b,color='black', linewidth=1)
    plt.show()

# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion