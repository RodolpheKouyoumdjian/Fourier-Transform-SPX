import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft
import yfinance as yf
import matplotlib.dates as mdates
from datetime import datetime
plt.rcParams['figure.figsize'] = [16,10]
plt.rcParams.update({'font.size':18})

# Download SPY data
data = yf.download("^GSPC", period="max")
f = np.array(data['Close'])
dates = data.index
n = len(f)
#plt.plot(t, f,color='c',label='Noisy')
#plt.legend()
#plt.show()

# Perform Fourier analysis
yf = rfft(f)
xf = rfftfreq(n, 1)
yf_abs = np.abs(yf)

mean = np.mean(yf_abs)
stdev = np.std(yf_abs)
indices = yf_abs > (mean + 0.2 * stdev) # filter out values
yf_clean = indices * yf # noise frequency will be set to 0
new_f_clean = irfft(yf_clean)

dtFmt = mdates.DateFormatter('%Y') # define the formatting
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.xticks(rotation=75, fontweight='light',  fontsize='x-small',)
plt.yticks(fontweight='light',  fontsize='x-small',)
plt.title("SPX price plotted against its Fourier Transform since 1950")
plt.gca().xaxis.set_major_locator(mdates.YearLocator())

try:
    plt.xlabel(xlabel="Year")
    plt.ylabel(ylabel="Price $USD")
    plt.plot(dates, new_f_clean, label="Clean")
    plt.plot(dates, f, linewidth=0.5, label="Noisy")
    plt.show()
except ValueError:
    plt.xlabel(xlabel="Year")
    plt.ylabel(ylabel="Price $USD")
    plt.plot(dates[1:], new_f_clean, label="Clean")
    plt.plot(dates, f, linewidth=0.5, label="Noisy")
    plt.show()
