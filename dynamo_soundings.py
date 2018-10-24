import numpy as np
import xarray
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import os
import seaborn as sns
sns.reset_orig()

UNITS = {'temp':'C', 'dewpoint':'C', 'pressure':'hPa', 'height':'m', 'RH':'%',
         'wnd_spd':'ms$^{-1}$', 'wnd_dir':'deg', 'u_wnd':'ms$^{-1}$', 'v_wnd':'ms$^{-1}$',}

PRIORITY_METADATA = {
     41112:  {'lat':18.23,  'lon':42.65,   'name':'Abha',                'elev':2090,   'n':324},
     41217:  {'lat':24.43,  'lon':54.65,   'name':'Abu Duabi',           'elev':27,     'n':345},
     43003:  {'lat':19.12,  'lon':72.85,   'name':'BomBay',              'elev':np.nan, 'n':118},
     43128:  {'lat':17.45,  'lon':78.47,   'name':'Hyderabad',           'elev':545,    'n':57},
     43279:  {'lat':13.00,  'lon':80.18,   'name':'Madras',              'elev':16,     'n':167},
     43285:  {'lat':12.95,  'lon':74.83,   'name':'Mangalore',           'elev':16,     'n':119},
     43333:  {'lat':11.67,  'lon':92.72,   'name':'Port Blair',          'elev':79,     'n':150},
     43369:  {'lat':8.30,   'lon':73.15,   'name':'Minacoy',             'elev':2,      'n':69},
     43371:  {'lat':8.48,   'lon':76.95,   'name':'Thiruvan',            'elev':64,     'n':121},
     48327:  {'lat':18.78,  'lon':98.98,   'name':'Chiang Mai Intl.',    'elev':323,    'n':126},
     48601:  {'lat':5.30,   'lon':100.27,  'name':'Penang/Bayan Lepas',  'elev':3,      'n':191},
     48820:  {'lat':21.02,  'lon':105.80,  'name':'Hanoi/Noibai Intl.',  'elev':9,      'n':349},
     48855:  {'lat':16.05,  'lon':108.20,  'name':'Da Nang',             'elev':7,      'n':357},
     48900:  {'lat':10.82,  'lon':106.67,  'name':'Saigon/Tan-Son-Nhut', 'elev':9,      'n':356},
     67083:  {'lat':-18.80, 'lon':47.48,   'name':'Antananarivo',        'elev':1,      'n':346},
     94150:  {'lat':-12.28, 'lon':136.82,  'name':'Gove',                'elev':53,     'n':167},
     94203:  {'lat':-17.95, 'lon':122.23,  'name':'Broome',              'elev':9,      'n':187},
     94294:  {'lat':-19.25, 'lon':146.77,  'name':'Townsville',          'elev':9,      'n':211},
     96315:  {'lat':4.93,   'lon':114.93,  'name':'Brunei Airport',      'elev':15,     'n':268},
     96413:  {'lat':1.48,   'lon':110.33,  'name':'Kuching',             'elev':27,     'n':213},
     96441:  {'lat':3.20,   'lon':113.03,  'name':'Bintulu',             'elev':5,      'n':212},
     96481:  {'lat':4.27,   'lon':117.83,  'name':'Tawau',               'elev':20,     'n':200},
     96996:  {'lat':-12.18, 'lon':96.82,   'name':'Cocos Island',        'elev':3,      'n':181},
     98223:  {'lat':18.18,  'lon':120.53,  'name':'Laoag',               'elev':4,      'n':244},
}

HIRES_METADATA = {
    96035:   {'lat':3.57,   'lon':98.68,   'name':'Medan',               'elev':27,     'n':np.nan},
    96163:   {'lat':-0.88,  'lon':100.35,  'name':'Padang',              'elev':2,      'n':np.nan},
    96237:   {'lat':-2.16,  'lon':106.13,  'name':'Pangkal',             'elev':33,     'n':np.nan},
    96749:   {'lat':-6.12,  'lon':106.68,  'name':'Jakarta',             'elev':10,     'n':np.nan},
    96935:   {'lat':-7.37,  'lon':112.78,  'name':'Surabaya',            'elev':3,      'n':np.nan},
    97014:   {'lat':1.54,   'lon':124.92,  'name':'Menado',              'elev':80,     'n':np.nan},
    97072:   {'lat':-0.91,  'lon':119.90,  'name':'Palu',                'elev':84,     'n':np.nan},
    97180:   {'lat':-5.06,  'lon':119.53,  'name':'Makassar',            'elev':12,     'n':np.nan},
    97372:   {'lat':-10.10, 'lon':123.66,  'name':'Kupang',              'elev':137,    'n':np.nan},
    97560:   {'lat':-1.19,  'lon':136.10,  'name':'Biak',                'elev':10,     'n':np.nan},
    97724:   {'lat':-3.71,  'lon':128.09,  'name':'Ambon',               'elev':10,     'n':np.nan},
    97980:   {'lat':-8.51,  'lon':140.4,   'name':'Merauke',             'elev':3,      'n':np.nan},
}

def load_dynamo_soundings(data_dir, priority=True, idate=None, fdate=None, 
                          interp=True, as_xarray=True, sixhourly=True, verbose=True):
    if as_xarray: assert interp
        
    if priority:
        if verbose and priority: print('Loading priority soundings...')
        elif verbose: print('Loading hi-resolution soundings...')
        metadata = PRIORITY_METADATA
        prefix = 'upaqf.'
    else:
        if verbose: print('Loading hi-res soundings...')
        metadata = HIRES_METADATA
        prefix = 'upaqi_'
    soundings_all = {}
    for wmo_id in metadata.keys():
        if verbose: print('  {} ({})'.format(metadata[wmo_id]['name'], wmo_id), end='')
        lines = open(os.path.join(data_dir, '{}{}'.format(prefix, wmo_id))).readlines()
        
        sounding_list = []
        for i, line in enumerate(lines):
            if 'STN' in line.split()[0]:
                if len(lines[i+1].split()) == 7:
                    _, yymmdd, hhmm, _, _, _, _ = lines[i+1].split()
                else:
                    yymmdd, hhmm, _, _, _, _ = lines[i+1].split()
                date = datetime.strptime('20'+yymmdd+hhmm, '%Y%m%d%H%M')
                if sixhourly and date.hour%6 != 0:
                    continue
                if idate is not None and date < idate:
                    continue
                if fdate is not None and date > fdate:
                    continue
                start_ind = i + 4
                end_ind = start_ind + 1
                if end_ind >= len(lines): continue
                while 'STN' not in lines[end_ind].split()[0] and end_ind < len(lines)-1:
                    end_ind += 1
        
                with open(os.path.join(data_dir, '{}{}'.format(prefix, wmo_id))) as fin:
                    reader = csv.reader(fin)
                    data = [[float(s) for s in row[0].split()] for i,row in enumerate(reader) if i in range(start_ind, end_ind)]
                sounding = SoundingOneTime(np.array(data)[:,:6], date, wmo_id, metadata[wmo_id])
                if interp:
                    sounding = sounding.interp_p()
                sounding_list.append(sounding)
        soundings_onestation = Sounding(sounding_list)
        if as_xarray:
            soundings_onestation = soundings_onestation.as_xarray()
        soundings_all[wmo_id] = soundings_onestation
        if verbose: print(' -- {} soundings'.format(soundings_onestation.ndates()))
    return AllSoundings(soundings_all, as_xarray=as_xarray)
                    

def get_mpas_soundings(mpas_data_dir, all_soundings, fcst_id='MPAS_3km', verbose=True):
    assert all_soundings.xarray
    
    if verbose: print('Getting {} soundings...'.format(fcst_id))
    for wmo_id, name in zip(all_soundings.wmo_ids, all_soundings.names):
        if verbose: print('  {} ({})'.format(name, wmo_id))
        sounding_xarray = all_soundings.soundings[wmo_id]
        
    return


class SoundingOneTime:
    
    def __init__(self, data, date, wmo_id, metadata, uv=False):
        assert data.shape[1] == 6
        self.pressure = data[:,0]
        self.height   = data[:,1]
        self.temp     = data[:,2]
        self.dewpoint = data[:,3]
        if uv:
            self.u  = data[:,4]
            self.v  = data[:,5]
        else:
            wnd_dir  = data[:,4]
            wnd_spd  = data[:,5]
            self.u = wnd_spd * np.cos(270. - wnd_dir)
            self.v = wnd_spd * np.sin(270. - wnd_dir)
        self.date     = date
        self.wmo_id   = wmo_id
        self.metadata = metadata
        self.name     = metadata['name']
        self.lat      = metadata['lat']
        self.lon      = metadata['lon']
        self.elev     = metadata['elev']
        
    def rh(self):
        t = self.temp
        td = self.dewpoint
        return 100.*(np.exp((17.625*td)/(243.04+td))/np.exp((17.625*t)/(243.04+t)))
    
    def interp_p(self, new_p=[950,925,900,875,850,825,800,750,700,650,600,500,400,300,200]):
        from scipy import interpolate
        newdata = np.zeros((len(new_p), 6))
        newdata[:, 0] = new_p
        old_p = self.pressure
        for d, data in enumerate([self.height, self.temp, self.dewpoint, self.u, self.v]):
            f = interpolate.interp1d(old_p, data, bounds_error=False)
            newdata[:, d+1] = f(new_p)
        return SoundingOneTime(newdata, self.date, self.wmo_id, self.metadata, uv=True)

class Sounding:
    
    def __init__(self, sounding_list):
        self.soundings = sounding_list
        self.dates     = np.array([s.date for s in self.soundings])
        self.wmo_id    = self.soundings[0].wmo_id
        self.name      = self.soundings[0].name
        self.lat       = self.soundings[0].lat
        self.lon       = self.soundings[0].lon
        self.elev      = self.soundings[0].elev
      
    def as_xarray(self):
        return SoundingXarray.from_sounding_obj(self)
    
    def nsoundings(self):
        return len(self.dates)
    def ndates(self):
        return len(self.dates)
        
    def get_sounding_bydate(self, date):
        ind = nearest_ind(self.dates, date)
        return self.soundings[ind]
    
    def plot_profiles(self, vrbl, ax=None, idate=None, fdate=None, pressure_coords=True, xlim=None, ylim=None,
                      showfig=True, fig_dir=None, set_labels=True, savefig=True, **plot_kwargs):
        assert vrbl in ['temp', 'dewpoint', 'wnd_spd', 'u_wnd', 'v_wnd', 'RH']
        
        XLIMS = {'temp':(-60, 30), 'dewpoint':(-80, 20), 'wnd_spd':(0,30),
                 'u_wnd':(-20,20), 'v_wnd':(-10,10), 'RH':(0,100)}
        
        if idate is None:
            idate = self.dates[0]
        if fdate is None:
            fdate = self.dates[-1]
        sns.set_style('ticks')
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        count = 0
        for s in self.soundings:
            if s.date < idate or s.date > fdate:
                continue
            if vrbl=='temp':       x = s.temp
            elif vrbl=='dewpoint': x = s.dewpoint
            elif vrbl=='RH':       x = s.rh()
            elif vrbl=='u_wnd':    x = s.u()
            elif vrbl=='v_wnd':    x = s.v()
            else:                  x = s.wnd_spd
            if pressure_coords:    y = s.pressure
            else:                  y = s.height
            ax.plot(x, y, **plot_kwargs)
            count += 1
        if pressure_coords:
            ylab = 'pressure'
            plt.gca().invert_yaxis()
            if ylim is None: ylim = (1000, 100)
        else:
            ylab = 'height'
            if ylim is None: ylim = (0, 12000)
        ax.grid('on')
        if xlim is None:
            ax.set_xlim(XLIMS[vrbl])
        ax.set_ylim(ylim)
        if set_labels:
            ax.set_xlabel('{} [{}]'.format(vrbl, UNITS[vrbl]))
            ax.set_ylabel('{} [{}]'.format(ylab, UNITS[ylab]))
            title = '{} ({}) soundings: {:%b%d}-{:%b%d}  (n={})'.format(self.name, self.wmo_id, idate, fdate, count)
            ax.set_title(title, loc='left')
        if savefig:
            if fig_dir is None:
                fig_dir = os.path.join(os.path.dirname(__file__), 'figures')
            if not os.path.isdir(fig_dir): os.makedirs(fig_dir)
            fig_file = os.path.join(fig_dir, '{}_{}_{:%b%d}-{:%b%d}.png'.format(self.wmo_id, vrbl, idate, fdate))
            plt.savefig(fig_file)
        if showfig: plt.show()
        else:       plt.close()
        sns.reset_orig()

        
        
class SoundingXarray(xarray.Dataset):
    
    @classmethod
    def from_sounding_obj(cls, sounding_obj):
        lengths = np.array([len(s.pressure) for s in sounding_obj.soundings])
        assert (lengths==lengths[0]).all()  # make sure all the soundings are on the same p levels
        
        # Create the coordinates and variables from the soundings
        coords = {
            'time': sounding_obj.dates,
            'pressure': sounding_obj.soundings[0].pressure,
        }
        vrbls = {
            'height': (('time', 'pressure'), np.array([s.height for s in sounding_obj.soundings])),
            'temp': (('time', 'pressure'), np.array([s.temp for s in sounding_obj.soundings])),
            'dewpoint': (('time', 'pressure'), np.array([s.dewpoint for s in sounding_obj.soundings])),
            'u': (('time', 'pressure'), np.array([s.u for s in sounding_obj.soundings])),
            'v': (('time', 'pressure'), np.array([s.v for s in sounding_obj.soundings]))
        }
        
        # Make the xarray Dataset object and set some attributes
        ds = xarray.Dataset(vrbls, coords)
        ds.attrs.update(wmo_id=sounding_obj.wmo_id, name=sounding_obj.name, lat=sounding_obj.lat,
                        lon=sounding_obj.lon, elev=sounding_obj.elev, type='obs')
        
        # Cast the Dataset as a SoundingXarray object and return
        ds.__class__ = cls
        return ds
    
    def nsoundings(self):
        return self.dims['time']
    def ndates(self):
        return self.dims['time']
        
    
class AllSoundings:
    
    def __init__(self, sounding_dict, as_xarray=False):
        self.soundings = sounding_dict
        self.wmo_ids   = list(self.soundings.keys())
        self.names     = [s.name for s in self.soundings.values()]
        self.lats      = np.array([s.lat for s in self.soundings.values()])
        self.lons      = np.array([s.lon for s in self.soundings.values()])
        self.elevs     = np.array([s.elev for s in self.soundings.values()])
        self.xarray    = as_xarray
    
    def nsites(self):
        return len(self.wmo_ids)
        
    def plot_sites_on_map(self, m, include_labels=True, real_names=True, **scatter_kwargs):
        if not include_labels: labels = None
        elif real_names:       labels = self.names
        else:                  labels = [str(wmo) for wmo in self.wmo_ids]
        
        X = []; Y = []
        for lat, lon in zip(self.lats, self.lons):
            x, y = m(lon, lat)
            X.append(x)
            Y.append(y)
        m.scatter(X, Y, **scatter_kwargs)
        if include_labels:
            for i, txt in enumerate(labels):
                plt.text(X[i], Y[i], txt)


        
def nearest_ind(array, value):
    return int((np.abs(array-value)).argmin())