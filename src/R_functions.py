# Some of the functions in this file has been copyed from https://towardsdatascience.com/how-to-write-your-favorite-r-functions-in-python-11e1e9c29089

def mean(array):
    """
    Calculates the mean of an array/vector
    """
    import numpy as np
    array=np.array(array)
    result= np.mean(array)
    return result

def sd(array):
    """
    Calculates the standard deviation of an array/vector
    """
    import statistics
    return statistics.stdev(array)

def median(array):
    """
    Calculates the median of an array/vector
    """
    import numpy as np
    array=np.array(array)
    result= np.median(array)
    return result

def var(array):
    """
    Calculates the variance of an array/vector
    """
    import statistics
    return statistics.variance(array)

def cov(x,y=None):
    """
    Calculates the covariance between two arrays/vectors or of a single matrix
    """
    import numpy as np
    array1=np.array(x)
    if y!=None:
        array2=np.array(y)
        if array1.shape!=array2.shape:
            print("Error: incompatible dimensions")
            return None
        covmat=np.cov(array1,array2)
        result=covmat[0][1]
    elif len(array1.shape)==1:
        result=float(np.cov(array1))
    else:
        result=np.cov(array1)
    return result

def cor(x, y):
    """
    Calculate Sample Correlation between two arrays
    """
    return (cov(x,y) / (sd(x) * sd(y)))

def dif(x, y, afrund = None):
    """
    Calculate the difference between two arrays
    """
    result = []

    if afrund == None:
        afrund = 2

    for index, element in enumerate(x):
        result.append(round((element - y[index]), afrund))

    return result

def summary(array):
    """
    Calculates the Tuckey Five-number (min/median/max/1st quartile/3rd quartile) of an array/vector
    """
    import numpy as np
    return {'min' : np.min(array), 'Q1' : np.percentile(array,25), 'median' : np.median(array), 'mean' : np.mean(array), 'Q3' : np.percentile(array,75), 'max' : np.max(array)}

def IQR(array):
    """
    Calculates the inter-quartile range of an array/vector
    """
    import numpy as np
    array=np.array(array)
    result = np.percentile(array,75)-np.percentile(array,25)
    
    return result

"""
Probability distributions
"""

#=====================
# Uniform distribution
#=====================

def dunif(x, minimum=0,maximum=1):
    """
    Calculates the point estimate of the uniform distribution
    """
    from scipy.stats import uniform
    result=uniform.pdf(x=x,loc=minimum,scale=maximum-minimum)
    return result

def punif(q, minimum=0,maximum=1):
    """
    Calculates the cumulative of the uniform distribution
    """
    from scipy.stats import uniform
    result=uniform.cdf(x=q,loc=minimum,scale=maximum-minimum)
    return result

def quantile(array):
    """
    Calculate and show R Quantile 0%, 25%, 50%, 75%, 100%
    """
    import numpy as np
    return {
        'Q0' : np.percentile(array, 0), 
        'Q1' : np.percentile(array, 25), 
        'Q2' : np.percentile(array, 50), 
        'Q3' : np.percentile(array, 75), 
        'Q4' : np.percentile(array, 100)
        }

def qunif(p, minimum=0,maximum=1):
    """
    Calculates the quantile function of the uniform distribution
    """
    from scipy.stats import uniform
    result=uniform.ppf(q=p,loc=minimum,scale=maximum-minimum)
    return result

def runif(n, minimum=0,maximum=1):
    """
    Generates random variables from the uniform distribution
    """
    from scipy.stats import uniform
    result=uniform.rvs(size=n,loc=minimum,scale=maximum-minimum)
    return result

#======================
# Binomial distribution
#======================

def dbinom(x,size,prob=0.5):
    """
    Calculates the point estimate of the binomial distribution
    """
    from scipy.stats import binom
    result=binom.pmf(k=x,n=size,p=prob,loc=0)
    return result

def pbinom(q,size,prob=0.5):
    """
    Calculates the cumulative of the binomial distribution
    """
    from scipy.stats import binom
    result=binom.cdf(k=q,n=size,p=prob,loc=0)
    return result

def qbinom(p, size, prob=0.5):
    """
    Calculates the quantile function from the binomial distribution
    """
    from scipy.stats import binom
    result=binom.ppf(q=p,n=size,p=prob,loc=0)
    return result

def rbinom(n,size,prob=0.5):
    """
    Generates random variables from the binomial distribution
    """
    from scipy.stats import binom
    result=binom.rvs(n=size,p=prob,size=n)
    return result

#=====================
# Normal distribution
#=====================

def dnorm(x,mean=0,sd =1):
    """
    Calculates the density of the Normal distribution
    """
    from scipy.stats import norm
    result=norm.pdf(x,loc=mean,scale=sd)
    return result

def pnorm(q,mean=0,sd=1):
    """
    Calculates the cumulative of the normal distribution
    """
    from scipy.stats import norm
    result=norm.cdf(x=q,loc=mean,scale=sd)
    return result

def qnorm(p,mean=0,sd=1):
    """
    Calculates the quantile function of the normal distribution
    """
    from scipy.stats import norm
    result=norm.ppf(q=p,loc=mean,scale=sd)
    return result

def rnorm(n,mean=0,sd=1):
    """
    Generates random variables from the normal distribution
    """
    from scipy.stats import norm
    result=norm.rvs(size=n,loc=mean,scale=sd)
    return result

#=====================
# Poisson distribution
#=====================

def dpois(x,mu):
    """
    Calculates the density/point estimate of the Poisson distribution
    """
    from scipy.stats import poisson
    result=poisson.pmf(k=x,mu=mu)
    return result

def ppois(q,mu):
    """
    Calculates the cumulative of the Poisson distribution
    """
    from scipy.stats import poisson
    result=poisson.cdf(k=q,mu=mu)
    return result

def qpois(p,mu):
    """
    Calculates the quantile function of the Poisson distribution
    """
    from scipy.stats import poisson
    result=poisson.ppf(q=p,mu=mu)
    return result

def rpois(n,mu):
    """
    Generates random variables from the Poisson distribution
    """
    from scipy.stats import poisson
    result=poisson.rvs(size=n,mu=mu)
    return result

#=====================
# chi^2-distribution
#=====================

def dchisq(x,df,ncp=0):
    """
    Calculates the density/point estimate of the chi-square distribution
    """
    from scipy.stats import chi2,ncx2
    if ncp==0:
        result=chi2.pdf(x=x,df=df,loc=0,scale=1)
    else:
        result=ncx2.pdf(x=x,df=df,nc=ncp,loc=0,scale=1)
    return result

def pchisq(q,df,ncp=0):
    """
    Calculates the cumulative of the chi-square distribution
    """
    from scipy.stats import chi2,ncx2
    if ncp==0:
        result=chi2.cdf(x=q,df=df,loc=0,scale=1)
    else:
        result=ncx2.cdf(x=q,df=df,nc=ncp,loc=0,scale=1)
    return result

def qchisq(p,df,ncp=0):
    """
    Calculates the quantile function of the chi-square distribution
    """
    from scipy.stats import chi2,ncx2
    if ncp==0:
        result=chi2.ppf(q=p,df=df,loc=0,scale=1)
    else:
        result=ncx2.ppf(q=p,df=df,nc=ncp,loc=0,scale=1)
    return result

def rchisq(n,df,ncp=0):
    """
    Generates random variables from the chi-square distribution
    """
    from scipy.stats import chi2,ncx2
    if ncp==0:
        result=chi2.rvs(size=n,df=df,loc=0,scale=1)
    else:
        result=ncx2.rvs(size=n,df=df,nc=ncp,loc=0,scale=1)
    return result

#==============================
# ### Student's t-distribution
#==============================

def dt(x,df,ncp=0):
    """
    Calculates the density/point estimate of the t-distribution
    """
    from scipy.stats import t,nct
    if ncp==0:
        result=t.pdf(x=x,df=df,loc=0,scale=1)
    else:
        result=nct.pdf(x=x,df=df,nc=ncp,loc=0,scale=1)
    return result

def pt(q,df,ncp=0):
    """
    Calculates the cumulative of the t-distribution
    """
    from scipy.stats import t,nct
    if ncp==0:
        result=t.cdf(x=q,df=df,loc=0,scale=1)
    else:
        result=nct.cdf(x=q,df=df,nc=ncp,loc=0,scale=1)
    return result

def qt(p,df,ncp=0):
    """
    Calculates the quantile function of the t-distribution
    """
    from scipy.stats import t,nct
    if ncp==0:
        result=t.ppf(q=p,df=df,loc=0,scale=1)
    else:
        result=nct.ppf(q=p,df=df,nc=ncp,loc=0,scale=1)
    return result

def rt(n,df,ncp=0):
    """
    Generates random variables from the t-distribution
    """
    from scipy.stats import t,nct
    if ncp==0:
        result=t.rvs(size=n,df=df,loc=0,scale=1)
    else:
        result=nct.rvs(size=n,df=df,nc=ncp,loc=0,scale=1)
    return result

#================
# F-distribution
#================

def df(x,df1,df2,ncp=0):
    """
    Calculates the density/point estimate of the F-distribution
    """
    from scipy.stats import f,ncf
    if ncp==0:
        result=f.pdf(x=x,dfn=df1,dfd=df2,loc=0,scale=1)
    else:
        result=ncf.pdf(x=x,dfn=df1,dfd=df2,nc=ncp,loc=0,scale=1)
    return result

def pf(q,df1,df2,ncp=0):
    """
    Calculates the cumulative of the F-distribution
    """
    from scipy.stats import f,ncf
    if ncp==0:
        result=f.cdf(x=q,dfn=df1,dfd=df2,loc=0,scale=1)
    else:
        result=ncf.cdf(x=q,dfn=df1,dfd=df2,nc=ncp,loc=0,scale=1)
    return result

def qf(p,df1,df2,ncp=0):
    """
    Calculates the quantile function of the F-distribution
    """
    from scipy.stats import f,ncf
    if ncp==0:
        result=f.ppf(q=p,dfn=df1,dfd=df2,loc=0,scale=1)
    else:
        result=ncf.ppf(q=p,dfn=df1,dfd=df2,nc=ncp,loc=0,scale=1)
    return result

def rf(n,df1,df2,ncp=0):
    """
    Calculates the quantile function of the F-distribution
    """
    from scipy.stats import f,ncf
    if ncp==0:
        result=f.rvs(size=n,dfn=df1,dfd=df2,loc=0,scale=1)
    else:
        result=ncf.rvs(size=n,dfn=df1,dfd=df2,nc=ncp,loc=0,scale=1)
    return result

#===================
# Beta distribution
#===================

def dbeta(x,shape1,shape2):
    """
    Calculates the density/point estimate of the Beta-distribution
    """
    from scipy.stats import beta
    result=beta.pdf(x=x,a=shape1,b=shape2,loc=0,scale=1)
    return result

def pbeta(q,shape1,shape2):
    """
    Calculates the cumulative of the Beta-distribution
    """
    from scipy.stats import beta
    result=beta.cdf(x=q,a=shape1,b=shape2,loc=0,scale=1)
    return result

def qbeta(p,shape1,shape2):
    """
    Calculates the cumulative of the Beta-distribution
    """
    from scipy.stats import beta
    result=beta.ppf(q=p,a=shape1,b=shape2,loc=0,scale=1)
    return result

def rbeta(n,shape1,shape2):
    """
    Calculates the cumulative of the Beta-distribution
    """
    from scipy.stats import beta
    result=beta.rvs(size=n,a=shape1,b=shape2,loc=0,scale=1)
    return result

#========================
# ### Gamma distribution
#========================

def dgamma(x,shape,rate=1):
    """
    Calculates the density/point estimate of the Gamma-distribution
    """
    from scipy.stats import gamma
    result=rate*gamma.pdf(x=rate*x,a=shape,loc=0,scale=1)
    return result

def pgamma(q,shape,rate=1):
    """
    Calculates the cumulative of the Gamma-distribution
    """
    from scipy.stats import gamma
    result=gamma.cdf(x=rate*q,a=shape,loc=0,scale=1)
    return result

def qgamma(p,shape,rate=1):
    """
    Calculates the cumulative of the Gamma-distribution
    """
    from scipy.stats import gamma
    result=(1/rate)*gamma.ppf(q=p,a=shape,loc=0,scale=1)
    return result

def rgamma(n,shape,rate=1):
    """
    Calculates the cumulative of the Gamma-distribution
    """
    from scipy.stats import gamma
    result=gamma.rvs(size=n,a=shape,loc=0,scale=1)
    return result