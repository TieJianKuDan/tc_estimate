from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import numpy as np
from matplotlib import pyplot as plt
from pandas import to_datetime


def dt64todt(dt64):
    ts = to_datetime(dt64).timestamp()
    return datetime.fromtimestamp(ts)

def geo_plot(lon, lat, data, levels=None):
    fig = plt.figure(figsize=(5, 5))
    proj = ccrs.PlateCarree(central_longitude=180)
    axe = plt.axes(projection=proj)
    axe.gridlines(
        draw_labels=True, dms=True, 
        x_inline=False, y_inline=False
    )
    axe.coastlines()
    axe.add_feature(cfeature.OCEAN)
    axe.add_feature(cfeature.LAND, edgecolor='b')

    lon = np.array([ele - 180 if ele > 0 else ele + 180 for ele in lon])
    contours = axe.contourf(
        lon, lat, data, 
        levels=levels,
        extend="both",
        transform=proj
    )
    fig.colorbar(contours, shrink=0.6, pad=0.15)
    plt.tight_layout()
    return fig

def tc_plot(lon, lat, data):
    levels = np.linspace(200, 310, 12)
    return geo_plot(lon, lat, data, levels)

def tc_gif(save_path, tc, fps=1, show_date=True):
    plt.ioff()
    imgs = [None] * len(tc)
    for i in range(len(tc)):
        fig = tc_plot(
            lon=tc[i].lon.data,
            lat=tc[i].lat.data,
            data=tc[i].data[0]
        )
        if not show_date:
            fig.suptitle(i)
        else:
            fig.suptitle(
                dt64todt(
                    tc[i].htime.data[0]
                ).strftime(r"%Y-%m-%d %H")
            )
        fig.canvas.draw()  
        image = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8)  
        image = image.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))  
        imgs[i] = image 
        plt.close(fig)
    
    imageio.mimsave(save_path, imgs, fps=fps)
