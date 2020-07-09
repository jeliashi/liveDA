from datetime import datetime, timedelta
import numpy as np
import xarray as xr

"""
'+proj=lcc +lat_0=38.5 +lon_0=262.5 +lat_1=38.5 +lat_2=38.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs'
Origin = (-2699020.142521930392832,1588193.847443336388096)
Pixel Size = (3000.000000000000000,-3000.000000000000000)
Corner Coordinates:
Upper Left  (-2699020.143, 1588193.847) (134d 7'17.14"W, 47d50'44.64"N)
Lower Left  (-2699020.143,-1588806.153) (122d43'44.80"W, 21d 7'19.89"N)
Upper Right ( 2697979.857, 1588193.847) ( 60d53'28.48"W, 47d50'57.51"N)
Lower Right ( 2697979.857,-1588806.153) ( 72d16'48.48"W, 21d 7'28.62"N)
Center      (    -520.143,    -306.153) ( 97d30'21.52"W, 38d29'50.09"N)
"""

class LC:
    def __init__(self, lat0, lon0, lat1, lat2, ellip='WGS84'):
        if ellip == 'WGS84':
            self.a = 6378137
            self.f = 1/298.257223563
            self.e = np.sqrt(self.f*2 - self.f**2)
        if lat1 == lat2:
            self.n = np.sin(lat1)
        else:
            self.n = np.log(self.m(lat1)/self.m(lat2))/np.log(self.t(lat1)/self.t(lat2))
        self.F = self.m(lat1)/(self.n * np.power(self.t(lat1), self.n))
    
        self.lat0 = lat0
        self.lon0 = lon0
    
    def m(self, phi): 
        return np.cos(phi)/np.sqrt(1-self.e**2 * np.sin(phi)**2)
    
    def t(self, phi):
        return np.tan(np.pi/4 - phi/2)/np.power((1-self.e*np.sin(phi))/(1+self.e*np.sin(phi)), self.e/2)
    
    def rho(self, lat):
        return self.a*self.F*np.power(self.t(lat), self.n)
    
    def __call__(self, lat,lon):
        gamma = self.n * (lon - self.lon0)
        y = self.rho(self.lat0) - self.rho(lat)*np.cos(gamma)
        x = self.rho(lat)*np.sin(gamma)
        return y, x

    def xy_to_latlon(self, x, y):
        rho_prime = np.sqrt(x**2 + (self.rho(self.lat0) - y)**2)
        t_prime = np.power(rho_prime/(self.F * self.a), 1/self.n)
        gamma_prime = np.arctan(x/(self.rho(self.lat0) - y))
        lat_prime = np.pi/2 - 2*np.arctan(t_prime)
        lat = np.pi/2 - 2*np.arctan(t_prime*np.power((1-self.e*np.sin(lat_prime))/(1+self.e*np.sin(lat_prime)), self.e/2))
        lon = gamma_prime/self.n + self.lon0
        return lat, lon

hrrr_LC = LC(38.5*np.pi/180., -97.5*np.pi/180., 38.5*np.pi/180., 38.5*np.pi/180.)

def forecast_to_time(da):
    init_time = datetime.strptime(da.initial_time, '%m/%d/%Y (%H:%M)') 
    ftime = timedelta(**{da.forecast_time_units:da.forecast_time.item()})
    return init_time + ftime

class Data:

    def __init__(self, data1, data2, crs=hrrr_LC):
        self.data1 = data1
        self.data2 = data2
        self.t1 = forecast_to_time(data1)
        self.t2 = forecast_to_time(data2)
        self.crs = crs
        self.ysize = data1.ygrid_0.size
        self.xsize = data1.xgrid_0.size

    @classmethod
    def chunk(cls, da):
        dtype = 'float32'
        assert da.dtype == dtype
        base_size = 2**21 # 2 MB
        bytes_per_val = 4 # float 32 is 4 bytes
        blocksize = base_size / bytes_per_val

        dimsizes = {d: da[d].size for d in da.dims}
        dimsize_order = sorted(dimsizes)
        chunksizes = {}
        for j in range(da.ndims+1, 0, -1):
            eq_chunk = int(blocksize ** (1/j))
            lowest_dim = dimsize_order.pop(0)
            nsize = min(eq_chunk, dimsizes[lowest_dim])
            chunksizes[lowest_dim] = nsize
            blocksize /= nsize
        
        return da.load.chunk(chunksizes)


    @classmethod
    def from_grib(cls, f1, f2, dvar='TMP_P0_L1_GLC0'):
        return cls(
            *list(map(
                lambda f: 
                cls.chunk(xr.open_dataset(f, engine='pynio')[dvar]),
                [f1,f2]
            ))
        )
    
    def latlong_to_xy(self, lat, lon):
        # pixel size is 3km
        y,x = self.crs(lat*np.pi/180.,lon*np.pi/180.)
        gridy = y / 3e3 + self.ysize // 2 + 1
        gridx = x / 3e3 + self.xsize // 2 + 1
        assert gridy >= 0
        assert gridy < self.ysize
        assert gridx >= 0
        assert gridx < self.xsize
        return {'gridlat_0': gridy, 'gridlon_0': gridx}
    
    def get_time_weights(self, time):
        assert time >= self.t1
        assert time <= self.t2
        dt = self.t2 - self.t1
        w2 = (time - self.t1)/dt
        return 1 - w2, w2

    def call_pt_at_time(self, time, lat, lon):
        w1, w2 = self.get_time_weights(time)

        locs = self.latlong_to_xy(lat, lon)
        return self.data1.isel(**locs)*w1 + self.data2.isel(**locs)*w2
    
    def get_box_slices(self, ll_lat, ll_lon, ur_lat, ur_lon):
        ll = self.latlong_to_xy(ll_lat, ll_lon)
        ur = self.latlong_to_xy(ur_lat, ur_lon)
        return {d: slice(ll[d],ur[d]+1) for d in ['gridlat_0', 'gridlon_0']}

    def call_box_at_time(self, time, ll_lat, ll_lon, ur_lat, ur_lon):
        slices = self.get_box_slices(ll_lat, ll_lon, ur_lat, ur_lon)
        w1, w2 = self.get_time_weights(time)
        return w1*self.data1.isel(**slices) + w2*self.data2.isel(**slices)

class MutatableData(Data):

    def __init__(self, *xargs, **kwargs):
        super().__init__(*xargs, **kwargs)
        self.bg = np.ones((2, self.ysize, self.xsize), dtype='float32')
    
    def put(self, val, errormax, loc_time, loc_lon, loc_lat, length_scale):
        # assume time is always within bound and error of time is infinite
        bias = val - self.call_pt_at_time(loc_time, loc_lon, loc_lat)

        
