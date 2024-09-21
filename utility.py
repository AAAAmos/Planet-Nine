import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import lmfit
from lmfit.lineshapes import gaussian2d
from lmfit.models import ExpressionModel

def CrossGaussian2dfit(file, cname, fitrange=60, grid=100, weight=0.9):

    data = pd.read_csv(file)

    n1 = cname[0]
    n2 = cname[1]

    x = (data.RA-data[n1])
    y = (data.DEC-data[n2])

    data = data.loc[(x>-0.02)&(x<0.02)&(y>-0.02)&(y<0.02),
                ['RA', 'DEC', n1, n2]
                ]

    x = (data.RA-data[n1])*3600*np.cos(data.DEC*np.pi/180)
    y = (data.DEC-data[n2])*3600

    xedges = np.linspace(-fitrange, fitrange, grid)
    yedges = np.linspace(-fitrange, fitrange, grid)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    # Histogram does not follow Cartesian convention (see Notes),
    # therefore transpose H for visualization purposes.
    H = H.T

    z = H.flatten()

    X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
    xf = X.flatten()
    yf = Y.flatten()
    # error = np.sqrt(z+1)
    w = z**weight+0.1

    model = lmfit.models.Gaussian2dModel()
    params = model.guess(z, xf, yf)
    result = model.fit(z, x=xf, y=yf, params=params, weights=w/10)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)

    vmax = np.nanpercentile(Z, 99.9)

    ax = axs[0, 0]
    art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Data')

    ax = axs[0, 1]
    fit = model.func(X, Y, **result.best_values)
    art = ax.pcolor(X, Y, fit, vmin=0, vmax=vmax, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Fit')

    ax = axs[1, 0]
    fit = model.func(X, Y, **result.best_values)
    art = ax.pcolor(X, Y, Z-fit, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Data - Fit')

    ax = axs[1, 1]
    ax.scatter(
        x, y,
        s = 1,
        alpha=0.2
    )
    ax.set_title('Origin data points')

    for ax in axs.ravel():
        ax.set_xlabel('Delta RA [arcsec]')
        ax.set_ylabel('Delta DEC [arcsec]')
    
    plt.show()

    lmfit.report_fit(result)

def CrossmatchDisfit(file, cname, cname0=['RA','DEC'], fitrange=70, grid=101, weight=1, mode=2):
    
    """
    This function is used to fit the crossmatch distance distribution of two catalogs.
    
    Return: 2D fitting plot
    
    Parameters:
    file: str, the file path of the crossmatch result
    cname: list, the column name of the second catalogs, first catalog is RA and DEC
    fitrange: int, the range of the fitting plot in arcsec
    grid: int, the number of the grid in the fitting plot, need to be odd
    weight: float, the weight of the fitting to the data
    mode: int, the mode of the fitting
        1: 1D fitting
        2: 2D fitting with ra, dec as the x, y axis
        3: 2D fitting with only radius variable
    """

    if type(file) == str:
        data = pd.read_csv(file)

        n1 = cname[0]
        n2 = cname[1]
    else:
        data = file
        n1 = cname[0]
        n2 = cname[1]

    x = (data[cname0[0]]-data[n1])*3600*np.cos(data[cname0[1]]*np.pi/180)
    y = (data[cname0[1]]-data[n2])*3600
    
    # print(data[n1])
    # print(data[cname0[0]])
    
    if mode == 1:
        
        fig, ax = plt.subplots()
        
        data = {
            'ra': ax.hist(x, grid)[1],
            'rac': ax.hist(x, grid+1)[0],
            'dec': ax.hist(y, grid)[1],
            'decc': ax.hist(y, grid+1)[0]
        }
        
        plt.close(fig)
        
        xedges = np.linspace(-fitrange, fitrange, grid)
        yedges = np.linspace(-fitrange, fitrange, grid)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        H = H.T
        z = H.flatten()
        
        X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
        xf = X.flatten()
        yf = Y.flatten()
        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
        vmax = np.nanpercentile(Z, 99.9)

        dframe = pd.DataFrame(data=data)

        model = LorentzianModel()

        paramsx = model.guess(dframe['rac'], x=dframe['ra'])
        paramsy = model.guess(dframe['decc'], x=dframe['dec'])

        resultra = model.fit(dframe['rac'], paramsx, x=dframe['ra'])
        cen1x = resultra.values['center']
        sig1x = resultra.values['sigma']
        resultdec = model.fit(dframe['decc'], paramsy, x=dframe['dec'])
        cen1y = resultdec.values['center']
        sig1y = resultdec.values['sigma']
        
        fitx = model.func(dframe['ra'], **resultra.best_values)
        fity = model.func(dframe['dec'], **resultdec.best_values)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        plt.rcParams.update({'font.size': 15})
        # ax = axs[0]
        # art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        # plt.colorbar(art, ax=ax, label='z')
        # ell = Ellipse(
        #         (cen1x, cen1y),
        #         width = 3*sig1x,
        #         height = 3*sig1y,
        #         edgecolor = 'w',
        #         facecolor = 'none'
        #     )
        # ax.add_patch(ell)
        # ax.set_title('Histogram of Data')
        # ax.set_xlabel('Delta RA [arcsec]')
        # ax.set_ylabel('Delta DEC [arcsec]')

        ax = axs[0]
        ax.plot(dframe['ra'], fitx, label='fit gaussian')
        ax.plot(dframe['ra'], dframe['rac'], 
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('Center:{0:5.4f}, 1 Sigma:{1:5.3f}'.format(cen1x, sig1x))
        ax.set_xlabel('Delta RA [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()

        ax = axs[1]
        ax.plot(dframe['dec'], fity, label='fit gaussian')
        ax.plot(dframe['dec'], dframe['decc'], 
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('Center:{0:5.4f}, 1 Sigma:{1:5.3f}'.format(cen1y, sig1y))
        ax.set_xlabel('Delta DEC [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()
        
        fig.suptitle('AKARI-TSL3 x '+file.split('-')[1][:-6] + '  1D fitting')
        
        plt.show()
    
    if mode == 2:

        xedges = np.linspace(-fitrange, fitrange, grid)
        yedges = np.linspace(-fitrange, fitrange, grid)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        H = H.T
        z = H.flatten()

        X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
        xf = X.flatten()
        yf = Y.flatten()
        
        w = z**weight+0.1
        
        model = Gaussian2dModel()
        params = model.guess(z, xf, yf)
        result = model.fit(z, x=xf, y=yf, params=params, weights=w/10)
        Amp = result.values['amplitude']
        cenx = result.values['centerx']
        sigx = result.values['sigmax']
        ceny = result.values['centery']
        sigy = result.values['sigmay']
        
        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
        vmax = np.nanpercentile(Z, 99.9)
        
        fit = model.func(X, Y, **result.best_values)

        Zx = Z[int((grid+1)/2)]
        fitx = fit[int((grid+1)/2)]
        Zy = Z.T[int((grid+1)/2)]
        fity = fit.T[int((grid+1)/2)]

        fig, axs = plt.subplots(2, 2, figsize=(15, 13))
        
        plt.rcParams.update({'font.size': 15})
        # plt.rcParams.update({"tick.labelsize": 13})
        
        ax = axs[0, 0]
        art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(art, ax=ax, label='Data point Density')
        ell = Ellipse(
                (cenx, ceny),
                width = 3*sigx,
                height = 3*sigy,
                edgecolor = 'w',
                facecolor = 'none'
            )
        ax.add_patch(ell)
        ax.set_title('Histogram of Data')
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)

        ax = axs[0, 1]
        art = ax.pcolor(X, Y, Z-fit, shading='auto')
        plt.colorbar(art, ax=ax, label='Data point Density')
        ax.set_title('Residual')
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)

        ax = axs[1, 0]
        ax.plot(xedges[:grid-1], fitx, label='fit gaussian')
        ax.plot(xedges[:grid-1], Zx, 
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('y-axis slice, Center:{0:5.3f}, 1σ:{1:5.2f}'.format(cenx, sigx))
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()

        ax = axs[1, 1]
        ax.plot(yedges[:grid-1], fity, label='fit gaussian')
        ax.plot(yedges[:grid-1], Zy,
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('x-axis slice, Center:{0:5.3f}, 1σ:{1:5.2f}'.format(ceny, sigy))
        ax.set_xlabel('ΔDEC [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()

        fig.suptitle('AKARI-TSL3 x '+file.split('-')[1][:-4]+'  2D fitting')

        plt.show()
        
    if mode == 3:

        xedges = yedges = np.linspace(-fitrange, fitrange, grid)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        H = H.T
        z = H.flatten()

        X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
        xf, yf = X.flatten(), Y.flatten()
        
        model = ExpressionModel(
            'amp*exp(-((x-cx)**2 / (2*sig**2)) - ((y-cy)**2 / (2*sig**2)))',
            independent_vars=['x', 'y']
        )
        params = model.make_params(amp=100, sig=fitrange/100, cx=0, cy=0)
        
        w = z**weight
        result = model.fit(z, x=xf, y=yf, params=params, weights=w)
        Sigma = result.params['sig'].value
        Amp = result.params['amp'].value
        print(Amp, Sigma)
        
        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
        Zx = Z[int((grid+1)/2)]
        Zy = Z.T[int((grid+1)/2)]

        fig, axs=plt.subplots(1, 2, figsize=(15, 5), dpi=100)

        ax = axs[0]
        ax.plot(xedges[:grid-1], Zx, 
            marker='s', markersize=5, ls='', label='data points'
            )
        ax.plot(np.linspace(-fitrange, fitrange, 100),
            model.eval(result.params, x=np.linspace(-fitrange, fitrange, 100), y=0),
            label=f'fit gaussian, $\\sigma$={Sigma:.4f}')
        ax.set_ylim(0, max(Zx)*1.2)
        ax.set_title('y-axis slice')
        ax.set_xlabel('Separation [arcsec]')
        ax.legend()

        ax=axs[1]
        ax.plot(yedges[:grid-1], Zy, 
            marker='s', markersize=5, ls='', label='data points'
            )
        ax.plot(np.linspace(-fitrange, fitrange, 100),
            model.eval(result.params, x=0, y=np.linspace(-fitrange, fitrange, 100)),
            label=f'fit gaussian, $\\sigma$={Sigma:.4f}')
        ax.set_ylim(0, max(Zx)*1.2)
        ax.set_title('x-axis slice')
        ax.set_xlabel('Separation [arcsec]')
        ax.legend()
        
        plt.suptitle(file.split('/')[-1])
        
        plt.show()
        
        return Amp, Sigma