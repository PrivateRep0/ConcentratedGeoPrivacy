from os import makedirs
from os.path import isdir
import numpy as np
from glob import glob
from datetime import datetime


def deg_to_rad(deg):
    rad = deg/360.*2*np.pi
    return rad


def convert_coord(lat, long, long0=0, R=6371000.0):
    x = R*(long-long0)
    y = R*np.log(np.tan(0.25*np.pi+0.5*lat))
    return (x, y)


def Extract_Cab_Data(filename,b_clean=False,time_max=86400,sp_max=3000,R=6371000.0):
    data = np.genfromtxt(filename,names=None)
    trace = []
    ind_sorted = data[:,-1].argsort()
    
    if b_clean:
        n = len(ind_sorted)
        ind = ind_sorted[0]
        row = data[ind]
        prev_time = datetime.fromtimestamp(row[-1])
        (x, y) = convert_coord(deg_to_rad(row[0]),deg_to_rad(row[1]),R=R)
        trace.append((x, y))
        prev_loc = (x, y)
        for i in range(1,n):
            ind = ind_sorted[i]
            row = data[ind]
            curr_time = datetime.fromtimestamp(row[-1])
            time_delta = (curr_time - prev_time).seconds
            if time_delta <= time_max:
                (x, y) = convert_coord(deg_to_rad(row[0]),deg_to_rad(row[1]),R=R)
                if np.sqrt((x-prev_loc[0])**2+(y-prev_loc[1])**2)/time_delta*60 <= sp_max:
                    trace.append((x, y))
                    prev_time = curr_time
                    prev_loc = (x, y)
    else:
        for ind in ind_sorted:
            row = data[ind]
            (x, y) = convert_coord(deg_to_rad(row[0]),deg_to_rad(row[1]),R=R)
            trace.append((x, y))
    return np.array(trace)


def Extract_Cab_Data_All(folder_in,len_min=5000,len_max=30000,R=6371000.0):
    dict_traj = {}
    trip_names = get_all_filenames(folder_in,'new_*.txt')
    for name in trip_names:
        x = Extract_Cab_Data(name,b_clean=True,R=R)
        n = len(x)
        if n < len_min or n > len_max:
            continue
        name_short = name.replace(folder_in,'').replace('/','').replace('\\','')
        dict_traj[name_short] = x
    return dict_traj


def get_all_filenames(dir,prefix=''):
    dirlist = glob(dir+'/'+prefix)
    return dirlist

    
def get_m_indices(n, m, b_cont=False, seed=None):
    if seed != None:
        np.random.seed(seed)
    indices = np.zeros(m)
    if m < n:
        if b_cont:
            l = int(m/2)
            r = m - l
            mid = np.random.randint(n)
            if mid-l >=0 and mid+r<n:
                start = mid-l
                indices[:l] = np.array([(start+i) for i in range(l)])
                indices[l:] = np.array([(mid+i) for i in range(r)])
            elif mid-l < 0:
                indices = np.array([i for i in range(m)])
            else:
                indices = np.array([(n-m+i) for i in range(m)])
        else:
            indices = np.random.randint(0,n,size=m)
    else:
        if b_cont:
            indices = np.arange(n)
        else:
            indices = np.random.permutation(n)
    return indices.astype(int)


def check_folder(folder):
    if not isdir(folder):
        makedirs(folder)


def update_visited_squares(tr, dict_visit):
    for point in tr:
        gr_x = int(np.floor(point[0]))
        gr_y = int(np.floor(point[1]))
        name = str(gr_x)+'_'+str(gr_y)
        if name in dict_visit.keys():
            dict_visit[name] = dict_visit[name]+1
        else:
            dict_visit[name] = 1


def get_visited_squares(path,R=6371000.0,suff='new_*.txt'):
    dict_visit = {}
    if suff == '':
        with open(path) as f:
            for line in f:
                (key, val) = line.split(' ')
                dict_visit[key] = int(val)
    else:
        trip_names = get_all_filenames(path,suff)
        for name in trip_names:
            x = Extract_Cab_Data(name,b_clean=True,sp_max=1500,R=R)#np.array(dict_trips[name])
            update_visited_squares(x,dict_visit)
    return dict_visit


def convert_visit_key(strkey):
    arrkey = strkey.split('_')
    x = float(arrkey[0])+0.5
    y = float(arrkey[1])+0.5
    return (x, y)


def jaccard_index_area(poly_np, poly_tilde):
    jacc = 0.0
    if poly_tilde.intersects(poly_np):
        poly_int = poly_tilde.intersection(poly_np)
        area_int = poly_int.area
        # jacc = area_int/(poly_np.area+poly_tilde.area-area_int)
        jacc = area_int/(poly_np.union(poly_tilde).area)
    return jacc

