##################################################################################
#
# StockPy Project
# Plot various statistics in real time of a stock portfolio
#
##################################################################################
import numpy as np
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pylab import *
import pandas as pd
from scipy.stats import norm
from scipy.optimize import leastsq
import matplotlib.mlab as mlab
import pandas_datareader.data as web
import warnings
import sys
import os

#matplotlib.style.use('ggplot') # makes grey backgrounds

fitfunc  = lambda p, x: p[0]*exp(-0.5*((x-p[1])/p[2])**2)+p[3]
errfunc  = lambda p, x, y: (y - fitfunc(p, x))

def symbol_to_csv_path(symbol, base_dir="."):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def symbol_to_svg_path(symbol, base_dir="."):
    """Return SVG file path given ticker symbol."""
    return os.path.join(base_dir, "{}.svg".format(str(symbol)))

def plot_daily_returns(df, stock, doplot, outdir=".", title="Daily Returns", xlabel="Date", ylabel="Stock Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    plt.savefig(symbol_to_svg_path(stock+'_daily',outdir))
    plt.title('$\mathrm{%s\ of\ %s\ }$' %(title, stock))
    if doplot == True:
        plt.show()
    plt.gcf().clear()
    plt.close()

def plot_daily_bollinger_bands(stockdf, stock, doplot, outdir, title="Bollinger Bands", xlabel="Date", ylabel="Price"):
    rm = get_rolling_mean(stockdf, 20)
    rstd = get_rolling_std(stockdf, 20)
    upper_band, lower_band = get_bollinger_bands(rm, rstd)
    ax = stockdf.plot(title=title, label=stock, color='Blue')
    hourFormatter = DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(hourFormatter)
    ax.set_xticklabels([], minor=True)
    rm.plot(label='Rolling mean', ax=ax, color='Gold')
    upper_band.plot(label='upper band', ax=ax, color='Red')
    lower_band.plot(label='lower band', ax=ax, color='Green')
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('$\mathrm{%s\ of\ %s\ }$' %(title, stock))
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, ['Current','Mean','Boll high','Boll low'])
    plt.savefig(symbol_to_svg_path(stock,outdir))
    if doplot == True:
        plt.show()
    plt.gcf().clear()
    plt.close()

def plot_daily_histogram(daily_returns, stock, index_to_compare, doplot, outdir, title="Histogram", xlabel="Date", ylabel="Price"):
    try:
        df1 = daily_returns[index_to_compare]/daily_returns[index_to_compare].max().astype(np.float64)
        df1.hist(bins=100,normed=1,label=index_to_compare)
        df2 = daily_returns[stock]/daily_returns[stock].max().astype(np.float64)
        df2.hist(bins=100,normed=1,label=stock)
    except:
        print 'skipping histogram plot of ', stock
    # show the mean and +- std on the plot
    mean = daily_returns[stock].mean()
    std = daily_returns[stock].std()
    axes = plt.gca()
    axes.axvline(mean,color='green',linestyle='dashed')
    axes.axvline(std,color='red',linestyle='dashed')
    axes.axvline(-std,color='red',linestyle='dashed')
    plt.legend(loc='upper right')
    labels = axes.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()

    mu = daily_returns[stock].skew()
    sigma = daily_returns[stock].kurt()
    """
    ydata    = daily_returns[stock].dropna().values
    xdata    = range(len(ydata))
    init  = [1.0, 0.5, 0.5, 0.5]

    out = leastsq(errfunc, init, args=(xdata, ydata))
    c = out[0]

    print "A exp[-0.5((x-mu)/sigma)^2] + k "
    print "Parent Coefficients:"
    print "1.000, 0.200, 0.300, 0.625"
    print "Fit Coefficients:"
    print c[0],c[1],abs(c[2]),c[3]
    
    df = pd.DataFrame(daily_returns[stock].copy()).dropna()
    fit = fitfunc(c, xdata)
    df[stock].ix[:] = fit
    #df.plot()
    """
    plt.title('$\mathrm{Histogram\ of\ %s:}\ \mu=%.3f,\ \sigma=%.3f$' %(stock, mu, sigma))
    plt.savefig(symbol_to_svg_path(stock+'_histogram',outdir))
    if doplot == True:
        plt.show()
    plt.gcf().clear()
    plt.close()

def plot_daily_scatter(daily_returns, stock, index_to_compare, doplot, outdir, title="Scatter", xlabel="Date", ylabel="Price"):
    daily_returns = daily_returns/daily_returns.max().astype(np.float64) # normalize
    beta_stock = 0.
    alpha_stock = 0.
    try:
        daily_returns.plot(kind='scatter',x=index_to_compare,y=stock)
        beta_stock, alpha_stock = np.polyfit(daily_returns[index_to_compare], daily_returns[stock], 1)
        plt.plot(daily_returns[index_to_compare], beta_stock*daily_returns[index_to_compare] + alpha_stock, '-', color='red')
    except:
        print 'skipping scatter plot of ', stock
 
    corr =  daily_returns.corr(method='pearson')
    plt.title(r'$\mathrm{%s\ of\ %s\ \alpha:\ %.2f\ \beta:\ %.2f\ corr:\ %.2f}\ $' %(title, stock, alpha_stock, beta_stock, corr[stock][index_to_compare]))
    plt.savefig(symbol_to_svg_path(stock+'_scatter',outdir))
    if doplot == True:
        plt.show()
    plt.gcf().clear()
    plt.close()
    # show the correlection between the stock and index

def plot_returns_history(stock, index_to_compare, start, end, doplot, outdir, title="YTD", xlabel="Date", ylabel="Price"):

    try:
        stockdata = web.DataReader(stock, "google", start, end)
        if pd.isnull(stockdata.values).all():
            raise 'Insufficient historical data '+stock
        indexdata = web.DataReader(index_to_compare, "google", start, end)
    except:
        return
    df = pd.DataFrame({stock: stockdata["Close"]})
    df2 = pd.DataFrame({stock: stockdata["Close"], index_to_compare: indexdata["Close"]})
    
    stock_change = df2.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
    stock_change.head()
    stock_change.plot(grid = True)

    plt.title(r'$\mathrm{%s\ Returns\ %s:}$' %(title, stock))
    plt.savefig(symbol_to_svg_path(stock+'_returns_history',outdir))
    if doplot == True:
        plt.show()
    plt.gcf().clear()
    plt.close()


def plot_history(stock, index_to_compare, start, end, doplot, outdir, title="YTD", xlabel="Date", ylabel="Price"):

    try:
        stockdata = web.DataReader(stock, "google", start, end)
        if pd.isnull(stockdata.values).all():
            raise 'Insufficient historical data '+stock
        indexdata = web.DataReader(index_to_compare, "google", start, end)
    except:
        return
    df = pd.DataFrame({stock: stockdata["Close"]})
    df2 = pd.DataFrame({stock: stockdata["Close"], index_to_compare: indexdata["Close"]})

    rm = get_rolling_mean(df, 20)
    rstd = get_rolling_std(df, 20)
    upper_band, lower_band = get_bollinger_bands(rm, rstd)
    ax = df.plot(title=title, grid = True, label=stock)
    rm.plot(label='Rolling mean', ax=ax, color='Gold')
    upper_band.plot(label='upper band', ax=ax, color='Red')
    lower_band.plot(label='lower band', ax=ax, color='Green')
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(r'$\mathrm{%s\ %s:}$' %(title, stock))
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, ['Current','Mean','Boll high','Boll low'])
    plt.savefig(symbol_to_svg_path(stock+'_history',outdir))
    
    if doplot == True:
        plt.show()
    plt.gcf().clear()
    plt.close()

def normalize_data(df):
    return df/df.ix[0,:]

def get_rolling_mean(df, window):
    """Return rolling mean of given values, using specified window size."""
    return df.rolling(window, win_type='boxcar').mean()

def get_rolling_std(df, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return df.rolling(window).std()

def get_bollinger_bands(rmean, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rmean+2.0*rstd
    lower_band = rmean-2.0*rstd
    return upper_band, lower_band

def compute_daily_returns(df):
    daily_returns = df.copy()
    #daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    #daily_returns.ix[0:] = 0.
    if len(daily_returns) > 0:
        daily_returns = df.pct_change(1)
        daily_returns.ix[0, :] = 0
    return daily_returns

def dostats(stock, index_to_compare, doplot=True, outdir="."):
    if stock == index_to_compare:
        return
    stockdf = pd.read_csv(symbol_to_csv_path(stock,outdir), index_col='Date')
    stockdf = pd.DataFrame(stockdf['Adj Close'])
    stockdf = stockdf.rename(columns={'Adj Close': stock})
    spydf = pd.read_csv(symbol_to_csv_path(index_to_compare,outdir), index_col='Date')
    spydf = pd.DataFrame(spydf['Adj Close'])
    spydf = spydf.rename(columns={'Adj Close': index_to_compare})
    df2   = pd.concat([stockdf.copy(), spydf.copy()])
    dfn   = normalize_data(df2.copy())

    stockdfj = pd.read_csv(symbol_to_csv_path(stock,outdir))
    stockdfj = pd.DataFrame(stockdfj['Adj Close'])
    stockdfj = stockdfj.rename(columns={'Adj Close': stock})
    spydfj = pd.read_csv(symbol_to_csv_path(index_to_compare,outdir))
    spydfj = pd.DataFrame(spydfj['Adj Close'])
    spydfj = spydfj.rename(columns={'Adj Close': index_to_compare})
    dfj   = stockdfj.join(spydfj.copy())
    dfj   = dfj.dropna(subset=[index_to_compare])
    #
    # do price plot, with Bollinger bands
    #
    plot_daily_bollinger_bands(stockdf, stock, doplot, outdir, title="Bollinger Bands", xlabel="Date", ylabel="Price")
    #
    # do stock daily return plot
    #
    daily_returns = compute_daily_returns(stockdf)
    plot_daily_returns(daily_returns, stock, doplot, outdir, title="Daily Returns", xlabel="Date", ylabel="Price")
    #
    # do histogram comparison plot against the index
    #
    daily_returns = compute_daily_returns(df2)
    plot_daily_histogram(daily_returns, stock, index_to_compare, doplot, outdir, title="Histogram", xlabel="Date", ylabel="Price")
    #
    # historical
    #
    year = datetime.date.today().strftime("%Y")
    start = datetime.datetime(int(year),1,1)
    end = datetime.date.today()
    plot_history(stock, index_to_compare, start, end, doplot, outdir, title="Historical", xlabel="Date", ylabel="Price")
    plot_returns_history(stock, index_to_compare, start, end, doplot, outdir, title="Historical Returns", xlabel="Date", ylabel="Price")
    #
    # do scatter plot
    #
    daily_returns = compute_daily_returns(dfj)
    plot_daily_scatter(daily_returns, stock, index_to_compare, doplot, outdir, title="Scatter Plot", xlabel="Date", ylabel="Price")


# "Symbol","Description","Quantity","Price","Price Change $","Price Change %","Market Value","Day Change $","Day Change %","Cost Basis","Gain/Loss $","Gain/Loss %","Reinvest Dividends?","Capital Gains?","% Of Account","Security Type",

def analyze_portfolio(symbols, schwab, index_to_compare, doplot=False, outdir="."):

    year = datetime.date.today().strftime("%Y")
    start = datetime.datetime(int(year),1,1)
    end = datetime.date.today()
    total_value = 0.0
    quantities = schwab['Quantity']
    quantities = quantities.apply(lambda x: x.replace(",",""))
    quantities = quantities.apply(lambda x: float(x))
    prices = schwab['Price']
    prices = prices.apply(lambda x: x.replace("$",""))
    prices = prices.apply(lambda x: float(x))
    market_value = schwab['Market Value']
    market_value = market_value.apply(lambda x: x.replace("$",""))
    market_value = market_value.apply(lambda x: x.replace(",",""))
    market_value = market_value.apply(lambda x: float(x))
    percents = schwab['% Of Account']
    percents = percents.apply(lambda x: x.replace("%",""))
    percents = percents.apply(lambda x: float(x))
    gain_loss_percent = schwab['Gain/Loss %']
    gain_loss_percent = gain_loss_percent.apply(lambda x: x.replace("%",""))
    gain_loss_percent = gain_loss_percent.apply(lambda x: float(x))
    values = prices * quantities
    total_value = values.sum()
    print "Total Portfolio Value ", "$%.2f" %total_value
    df1 = pd.DataFrame(market_value.values/total_value*1000.,index=symbols)
    df1.columns = ['Market Value']
    df3 = pd.DataFrame(gain_loss_percent.values,index=symbols)
    df3.columns = ['Gain/Loss %']
    df2 = df1.join(df3)
    ax = df2.plot(kind='bar', stacked=True)
    ax.set(xlabel="Stock Portfolio", ylabel="% Value")
    ax.set_xticklabels(symbols)
    rects = ax.patches
    labels = map(lambda x: "%.2f" % x, values)
    for rect, label in zip(rects, values):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', rotation=90)
    plt.title(r'$\mathrm{Portfolio\ Value\ \$%.2f\ }$' %(total_value))
    plt.savefig(symbol_to_svg_path('portfolio_bar',outdir))
    if doplot == True:
        plt.show()
    plt.gcf().clear()
    plt.close()

    try:
        warnings.simplefilter("ignore")
        stockdata = web.DataReader(symbols, "google", start, end)
        if pd.isnull(stockdata.values).all():
            raise 'Insufficient historical data '+stock
        indexdata = web.DataReader(index_to_compare, "google", start, end)
    except:
        return

    df = pd.DataFrame(stockdata["Close"])
    df = df.dropna(axis=1, how='all')
    dfn   = normalize_data(df)
    colors = cm.rainbow(np.linspace(0, 1, len(symbols)))
    ax = dfn.plot(title="Normalized Portfolio Plot", grid = True, legend=False, color=colors)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.savefig(symbol_to_svg_path('portfolio_history',outdir))
    if doplot == True:
        plt.show()
    plt.gcf().clear()
    plt.close()

    dfs = df.sum(axis=1)
    dfns = normalize_data(dfs)
    dfi = pd.DataFrame(indexdata["Close"])
    dfi = dfi.dropna(axis=1, how='all')
    dfni = normalize_data(dfi)
    dfni = get_rolling_mean(dfni, 20)
    dfns = get_rolling_mean(dfns, 20)
    df2   = pd.concat([dfns.copy(), dfni.copy()])
    ax = df2.plot(title="Comparative Portfolio Plot", grid = True)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, ['Portfolio','SPY'])
    plt.savefig(symbol_to_svg_path('portfolio_comp',outdir))
    if doplot == True:
        plt.show()
    plt.gcf().clear()
    plt.close()

##################################################################################
#
# main
#
##################################################################################

if __name__ == '__main__':
    doplot = False
    csvfile = ""
    outdir = "."

    if len(sys.argv) == 1:
        print argv[0], ": <Schwab export stock portfolio> <output directory> plot|noplot"
        exit()

    if len(sys.argv) >= 2:
        csvfile = sys.argv[1]

    if len(sys.argv) >= 3:
        outdir = sys.argv[2]
        if not os.access(outdir, os.R_OK):
            os.makedirs(outdir)

    if len(sys.argv) >= 4:
        doplot = sys.argv[3]=='plot'

    df = pd.read_csv(csvfile,skiprows=2) # read schwab positions downloads
    df = pd.DataFrame(df)[:-2]
    symbols = df['Symbol'].values
    index_to_compare = 'SPY'    # this is the baseline of all stocks traded, is this stock doing better or worse?
    for symbol in symbols:
        dostats(symbol, index_to_compare, doplot, outdir)

    analyze_portfolio(symbols, df, index_to_compare, doplot, outdir)

