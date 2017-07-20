##################################################################################
#
# StockPy Project
#
# get current stock quotes from google and analyze/plot with stats.py
#
##################################################################################

import pandas as pd
import numpy as np
import json
import sys
from urllib2 import Request, urlopen
import time
import os
import stats

def symbol_to_csv_path(symbol, base_dir="."):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

googleFinanceKeyToFullName = {
    'id'     : 'ID',
    't'      : 'StockSymbol',
    'e'      : 'Index',
    'l'      : 'LastTradePrice',
    'l_cur'  : 'LastTradeWithCurrency',
    'ltt'    : 'LastTradeTime',
    'lt_dts' : 'LastTradeDateTime',
    'lt'     : 'LastTradeDateTimeLong',
    'div'    : 'Dividend',
    'yld'    : 'Yield'
}

def buildUrl(symbols):
    symbol_list = ','.join([symbol for symbol in symbols])
    # a deprecated but still active & correct api
    return 'http://finance.google.com/finance/info?client=ig&q=' + symbol_list

def request(symbols):
    url = buildUrl(symbols)
    req = Request(url)
    resp = urlopen(req)
    # remove special symbols such as the pound symbol
    content = resp.read().decode('ascii', 'ignore').strip()
    content = content[3:]
    return content

def replaceKeys(quotes):
    global googleFinanceKeyToFullName
    quotesWithReadableKey = []
    for q in quotes:
        qReadableKey = {}
        for k in googleFinanceKeyToFullName:
            if k in q:
                qReadableKey[googleFinanceKeyToFullName[k]] = q[k]
        quotesWithReadableKey.append(qReadableKey)
    return quotesWithReadableKey

def getQuotes(symbols):
    '''
        quotes = getQuotes(['AAPL', 'GOOG'])
        return:
        [{'Index': 'NASDAQ', 'LastTradeWithCurrency': '129.09', 'LastTradeDateTime': '2015-03-02T16:04:29Z', 'LastTradePrice': '129.09', 'Yield': '1.46', 'LastTradeTime': '4:04PM EST', 'LastTradeDateTimeLong': 'Mar 2, 4:04PM EST', 'Dividend': '0.47', 'StockSymbol': 'AAPL', 'ID': '22144'}, {'Index': 'NASDAQ', 'LastTradeWithCurrency': '571.34', 'LastTradeDateTime': '2015-03-02T16:04:29Z', 'LastTradePrice': '571.34', 'Yield': '', 'LastTradeTime': '4:04PM EST', 'LastTradeDateTimeLong': 'Mar 2, 4:04PM EST', 'Dividend': '', 'StockSymbol': 'GOOG', 'ID': '304466804484872'}]
        
        :param symbols: a single symbol or a list of stock symbols
        :return: real-time quotes list
        '''
    if type(symbols) == type('str'):
        symbols = [symbols]
    content = json.loads(request(symbols))
    return replaceKeys(content);

def fetchPreMarket(symbols):
    """
        // [
            {
            "id": "22144"
            ,"t" : "AAPL"
            ,"e" : "NASDAQ"
            ,"l" : "125.80"
            ,"l_fix" : "125.80"
            ,"l_cur" : "125.80"
            ,"s": "1"
            ,"ltt":"4:02PM EDT"
            ,"lt" : "May 5, 4:02PM EDT"
            ,"lt_dts" : "2015-05-05T16:02:28Z"
            ,"c" : "-2.90"
            ,"c_fix" : "-2.90"
            ,"cp" : "-2.25"
            ,"cp_fix" : "-2.25"
            ,"ccol" : "chr"
            ,"pcls_fix" : "128.7"
            ,"el": "126.10"
            ,"el_fix": "126.10"
            ,"el_cur": "126.10"
            ,"elt" : "May 6, 6:35AM EDT"
            ,"ec" : "+0.30"
            ,"ec_fix" : "0.30"
            ,"ecp" : "0.24"
            ,"ecp_fix" : "0.24"
            ,"eccol" : "chg"
            ,"div" : "0.52"
            ,"yld" : "1.65"
            ,"eo" : ""
            ,"delay": ""
            ,"op" : "128.15"
            ,"hi" : "128.45"
            ,"lo" : "125.78"
            ,"vo" : "21,812.00"
            ,"avvo" : "46.81M"
            ,"hi52" : "134.54"
            ,"lo52" : "82.90"
            ,"mc" : "741.44B"
            ,"pe" : "15.55"
            ,"fwpe" : ""
            ,"beta" : "0.84"
            ,"eps" : "8.09"
            ,"shares" : "5.76B"
            ,"inst_own" : "62%"
            ,"name" : "Apple Inc."
            ,"type" : "Company"
            }
        ]
    """
    """
        def fetchPreMarket(symbol, exchange):
        link = "http://finance.google.com/finance/info?client=ig&q="
        url = link+"%s:%s" % (exchange, symbol)
        u = urllib2.urlopen(url)
        content = u.read()
        data = json.loads(content[3:])
        info = data[0]
        t = str(info["elt"])    # time stamp
        l = float(info["l"])    # close price (previous trading day)
        p = float(info["el"])   # stock price in pre-market (after-hours)
        return (t,l,p)
        
        
        p0 = 0
        while True:
        t, l, p = fetchPreMarket("AAPL","NASDAQ")
        if(p!=p0):
        p0 = p
        print("%s\t%.2f\t%.2f\t%+.2f\t%+.2f%%" % (t, l, p, p-l,
        (p/l-1)*100.))
        time.sleep(60)
    """
    symbol_list = ','.join([symbol for symbol in symbols])
    url = "http://finance.google.com/finance/info?infotype=infoquoteall&client=ig&q=%s" % (symbol_list)
    req = Request(url)
    resp = urlopen(req)
    content = resp.read().decode('ascii', 'ignore').strip()
    content = content[3:]
    content = json.loads(content)
    info = content[0]
    # the prefix e means extended hours
    t = ""
    p = 0.
    l = float(info["l"])        # close price (previous trading day)
    if info.has_key("elt"):
        t = str(info["elt"])    # time stamp
    if info.has_key("el"):
        p = float(info["el"])   # stock price in pre-market (after-hours)

    print("%s\t%.2f\t%.2f\t%+.2f\t%+.2f%%" % (t, l, p, p-l, (p/l-1)*100.))

    return info.has_key("el")

    """
    <?xml version="1.0" encoding="UTF-8"?>
    <response>
      <result type="getquotes" timestamp="1405539220">
        <list count="1" total="1">
          <quote>
            ...
            <!-- Verify that this is the exchange you are interested in -->
            <exchange>NYSE</exchange>
            ...
            <!-- 0 means market is closed, anything else means its open -->
            <status>1</status>
            ...
          </quote>
        </list>
      </result>
    </response>
    """

##################################################################################
#
# main
#
##################################################################################

if __name__ == '__main__':
    
    if len(sys.argv) == 1:
        print argv[0], ": <Schwab export stock portfolio> <output directory>"
        exit()

    csvfile = ""
    outdir = "."
    append = ""
    index_to_compare = 'SPY'

    if len(sys.argv) >= 2:
        csvfile = sys.argv[1]

    if len(sys.argv) >= 3:
        outdir = sys.argv[2]
        if not os.access(outdir, os.R_OK):
            os.makedirs(outdir)

    if len(sys.argv) >= 4:
        append = sys.argv[3]

    df = pd.read_csv(csvfile,skiprows=2) # read schwab positions downloads
    df = pd.DataFrame(df)
    symbols = df['Symbol'].values[:-2]
    if index_to_compare not in symbols:
        symbols = np.concatenate(([index_to_compare], symbols))

    if append != "-a":
        for i in range(len(symbols)):
            with open(symbol_to_csv_path(symbols[i], outdir), "w") as myfile:
                myfile.write('Date,Open,High,Low,Close,Volume,Adj Close\n')

    tick = 1
    price = 0.0
    openprice = 0.0
    closeprice = 0.0
    adjcloseprice = 0.0
    lowprice = 0.0
    highprice = 0.0
    volume = 0
    sleeptimesec = 60
    lastdatetime = ""
    dostatistics = False
    dopremarket = False
    doplot = False

    while True:
        quotes = getQuotes(symbols)
        markdatetime = quotes[0]['LastTradeDateTime'].replace("T", " ").replace("Z", "")
        if lastdatetime != "":
            if lastdatetime != markdatetime:
                for i in range(len(quotes)):
                    thedatetime = quotes[i]['LastTradeDateTime'].replace("T", " ").replace("Z", "")
                    price = float(quotes[i]['LastTradePrice'].replace(",",""))
                    closeprice = price
                    adjcloseprice = price
                    if tick == 1:
                        openprice = price
                        lowprice = price
                        highprice = price
                    if lowprice > price:
                        lowprice = price
                    if highprice < price:
                        highprice = price
                    volume = int(quotes[i]['ID'])
                    symbol = quotes[i]['StockSymbol']
                    with open(symbol_to_csv_path(symbol, outdir), "a") as myfile:
                        myfile.write(thedatetime+","+str(openprice)+","+str(highprice)+","+str(lowprice)+","+str(closeprice)+","+str(volume)+","+str(adjcloseprice)+'\n')
                    if dostatistics:
                        stats.dostats(symbol, index_to_compare, doplot, outdir)

                if dostatistics:
                    stats.analyze_portfolio(symbols, df, index_to_compare, doplot, outdir)

                if dopremarket:
                    fetchPreMarket(symbols)

        lastdatetime = markdatetime
        if lastdatetime != markdatetime:
            tick += 1
        time.sleep(sleeptimesec)
        #print(json.dumps(quotes, indent=2))


