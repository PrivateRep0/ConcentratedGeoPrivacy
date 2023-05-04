import numpy as np
from multiprocessing import Process
from utils import Extract_Cab_Data_All, get_visited_squares, convert_visit_key, check_folder, get_m_indices, jaccard_index_area
from GP.algo import GPBasic, GP_kPNN, GP_PCH_point_adaptk
from CGP.algo import CGPBasic, CGP_kPNN, CGP_PCH_point_adaptk
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

def test_traj_init(folder_out, rho, eps_delta, traj):
    trip_names = list(traj.keys())
    for name_short in trip_names:
        x = traj[name_short]
        filename_out = folder_out+'/test_traj_'+ name_short
        file = open(filename_out, 'w+')
        file.write('name;index_max_cgpbasic;index_max_gpbasic;dist_max_cgpbasic;dist_max_gpbasic;dist_l2_cgpbasic;dist_l2_gpbasic\n')
        #privatize each loc with cgp and gp
        yc = CGPBasic(x, rho)
        yg = GPBasic(x, eps_delta)
        #basic cgp composition
        tc = np.argmax(np.linalg.norm(yc-x,axis=1))
        di_c = np.linalg.norm(yc[tc]-x[tc])
        dc2 = np.linalg.norm(yc-x)
        #basic gp composition
        tg = np.argmax(np.linalg.norm(yg-x,axis=1))
        di_g = np.linalg.norm(yg[tg]-x[tg])
        dg2 = np.linalg.norm(yg-x)
        line = ';'.join([name_short, str(tc), str(tg), str(di_c), str(di_g), str(dc2), str(dg2)])
        file.write(line+'\n')
        file.close()
        

def test_kpnn_init(folder_out, rho, eps_delta, traj, p, k, gamm_fact=0.0):
    trip_names = list(traj.keys())
    for name_short in trip_names:
        x = traj[name_short]
        filename_out = folder_out+'/test_knn_'+ name_short
        file = open(filename_out, 'w+')
        file.write('name;point;index_nonprivate;index_pnn_cgp;index_pnn_gp;index_cgpbasic;index_gpbasic;dist_nonprivate;dist_pnn_cgp;dist_pnn_gp;dist_cgpbasic;dist_gpbasic\n')

        #non-private
        dists = np.linalg.norm(x-p,axis=1)
        Jn = np.argpartition(dists,k)[:k]
        #privatize each loc with cgp and gp
        yc = CGPBasic(x, rho)
        yg = GPBasic(x, eps_delta)
        dists_yc = np.linalg.norm(yc-p,axis=1)
        Jc = np.argpartition(dists_yc,k)
        dists_yg = np.linalg.norm(yg-p,axis=1)
        Jg = np.argpartition(dists_yg,k)
        ##private kpnn with cgp composition
        Jc_alg = CGP_kPNN(x, p, rho, k, gamm_fact=gamm_fact)
        ##private kpnn with gp composition
        Jg_alg = GP_kPNN(x, p, eps_delta, k, gamm_fact=gamm_fact)
        
        for j in range(k):
            tx = Jn[j]
            d_x = dists[tx]
            t = Jc_alg[j]
            di_t = dists[t]
            td = Jg_alg[j]
            di_d = dists[td]
            tc = Jc[j]
            di_c = dists[tc]
            tg = Jg[j]
            di_g = dists[tg]

            line = ';'.join([name_short, str(p), str(tx), str(t), str(td), str(tc), str(tg), str(d_x), str(di_t), str(di_d), str(di_c), str(di_g)])
            file.write(line+'\n')

        file.close() 
        

def test_convh_init(folder_out, rho, eps_delta, traj):
    trip_names = list(traj.keys())
    for name_short in trip_names:
        x = traj[name_short]
        filename_out = folder_out+'/test_convhpj_'+ name_short
        file = open(filename_out, 'w+')
        file.write('name;k_nonprivate;k_pch_cgp;k_pch_gp;k_cgpbasic;k_gpbasic;jacc_nonprivate;jacc_pch_cgp;jacc_pch_gp;jacc_cgpbasic;jacc_gpbasic;area_nonprivate;area_pch_cgp;area_pch_gp;area_cgpbasic;area_gpbasic\n')
        #nonprivate
        hull_np = ConvexHull(x)
        k_np = len(hull_np.vertices)
        poly_np = Polygon(x[hull_np.vertices])
        jacc_np = 1.0#jaccard_index_area(poly_np, poly_np)

        #privatize each loc with cgp and gp
        yc = CGPBasic(x, rho)
        yg = GPBasic(x, eps_delta)
        hull_c = ConvexHull(yc) 
        k_c = len(hull_c.vertices)
        poly_c = Polygon(yc[hull_c.vertices])
        jacc_c = jaccard_index_area(poly_np, poly_c)

        hull_g = ConvexHull(yg)
        k_g = len(hull_g.vertices)
        poly_g = Polygon(yg[hull_g.vertices])
        jacc_g = jaccard_index_area(poly_np, poly_g)

        #cgp pch
        y_A = CGP_PCH_point_adaptk(x,rho,beta=0.1,alpha=0.92)
        hull_cpch = ConvexHull(y_A)
        k_cpch = len(hull_cpch.vertices)
        poly_cpch = Polygon(y_A[hull_cpch.vertices])
        jacc_cpch = jaccard_index_area(poly_np, poly_cpch)
        #gp pch
        y_Ag= GP_PCH_point_adaptk(x,eps_delta,beta=0.1,alpha=0.92)
        hull_gpch = ConvexHull(y_Ag)
        k_gpch = len(hull_gpch.vertices)
        poly_gpch = Polygon(y_Ag[hull_gpch.vertices])
        jacc_gpch = jaccard_index_area(poly_np, poly_gpch)
        line = ';'.join([name_short, str(k_np), str(k_cpch), str(k_gpch), str(k_c), str(k_g), str(jacc_np), str(jacc_cpch), str(jacc_gpch), str(jacc_c), str(jacc_g), str(poly_np.area), str(poly_cpch.area), str(poly_gpch.area), str(poly_c.area), str(poly_g.area)])
        file.write(line+'\n')
        file.close()


def test_traj_m_single(folder_out_prefix, ms, rho, eps_delta, arr_dict_traj, seed, params, i):
    num_ms = len(ms)
    np.random.seed(seed)
    for j in range(num_ms):
        m = ms[j]    
        folder_out = folder_out_prefix+'test_traj_rho'+str(rho)+'_m'+str(m)+'_rep'+str(i)
        check_folder(folder_out)
        print(folder_out)
        filelog = folder_out+'/params.txt'
        with open(filelog,'w+') as log:
            log.write(params.replace('<folder_out>',folder_out).replace('<m>',str(m)).replace('<seed>',str(seed)))
        test_traj_init(folder_out, rho, eps_delta, arr_dict_traj[j])


def test_traj_m(folder_in, num_rep, ms, rho, eps_delta, seeds, run_range, folder_out_prefix='./results/', delta=1e-10, b_cont=False, b_mp=False, R=6371000.0):
    dict_traj = Extract_Cab_Data_All(folder_in,R=R)
    params = 'foder_in;folder_out;m_len;rho;delta;eps_delta;seed\n'
    params = params + folder_in+';<folder_out>;<m>;'+str(rho)+';'+str(delta)+';'+str(eps_delta)+';<seed>'
    if not(b_mp):
        for i in run_range:
            arr_dict_traj = [{} for m in ms]
            np.random.seed(seeds[i])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                for j in range(len(ms)):
                    m_len = ms[j]
                    indices = get_m_indices(n, m_len, b_cont=b_cont)
                    arr_dict_traj[j][key] = x[indices]
            test_traj_m_single(folder_out_prefix, ms, rho, eps_delta, arr_dict_traj, seeds[num_rep+i], params, i)
    else:
        procs = []
        for i in run_range:
            arr_dict_traj = [{} for m in ms]
            np.random.seed(seeds[i])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                for j in range(len(ms)):
                    m_len = ms[j]
                    indices = get_m_indices(n, m_len, b_cont=b_cont)
                    arr_dict_traj[j][key] = x[indices]
            procs.append(Process(target=test_traj_m_single,args=[folder_out_prefix, ms, rho, eps_delta, arr_dict_traj, seeds[num_rep+i], params, i]))
            procs[i-run_range[0]].start()
        for i in run_range:
            procs[i-run_range[0]].join()


def test_traj_rho_single(folder_out_prefix, m_len, rhos, eps_deltas, dict_traj, seed, params, i):
    num_rhos = len(rhos)
    np.random.seed(seed)
    for j in range(num_rhos):
        rho = rhos[j]
        eps_delta = eps_deltas[j]        
        folder_out = folder_out_prefix+'test_traj_rho'+str(rho)+'_m'+str(m_len)+'_rep'+str(i)
        check_folder(folder_out)
        print(folder_out)
        filelog = folder_out+'/params.txt'
        with open(filelog,'w+') as log:
            log.write(params.replace('<folder_out>',folder_out).replace('<rho>',str(rho)).replace('<eps_delta>',str(eps_delta)).replace('<seed>',str(seed)))
        test_traj_init(folder_out, rho, eps_delta, dict_traj)


def test_traj_rho(folder_in, num_rep, m_len, rhos, eps_deltas, seeds, run_range, folder_out_prefix='./results/', delta=1e-10, b_cont=False, b_mp=False, R=6371000.0):
    dict_traj = Extract_Cab_Data_All(folder_in,R=R)
    params = 'foder_in;folder_out;m_len;rho;delta;eps_delta;seed\n'
    params = params + folder_in+';<folder_out>;'+str(m_len)+';<rho>;'+str(delta)+';<eps_delta>;<seed>'
    if not(b_mp):
        for i in run_range:
            dict_traj_i = {}
            np.random.seed(seeds[i])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                indices = get_m_indices(n, m_len, b_cont=b_cont)
                dict_traj_i[key] = x[indices]
            test_traj_rho_single(folder_out_prefix, m_len, rhos, eps_deltas, dict_traj_i, seeds[num_rep+i], params, i)
    else:
        procs = []
        for i in run_range:
            dict_traj_i = {}
            np.random.seed(seeds[i])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                indices = get_m_indices(n, m_len, b_cont=b_cont)
                dict_traj_i[key] = x[indices]
            procs.append(Process(target=test_traj_rho_single,args=[folder_out_prefix, m_len, rhos, eps_deltas, dict_traj_i, seeds[num_rep+i], params, i]))
            procs[i-run_range[0]].start()
        for i in run_range:
            procs[i-run_range[0]].join()


def test_kpnn_m_single2(folder_out_prefix, ms, ks, rho, eps_delta, arr_dict_traj, p, seed, params, i):
    num_ms = len(ms)
    num_ks = len(ks)
    np.random.seed(seed)
    for k in range(num_ks):
        k_nn = ks[k]
        if k_nn >= 125:
            continue
        for j in range(num_ms):
            m = ms[j]     
            folder_out = folder_out_prefix+'test_kpnn2_rho'+str(rho)+'_k'+str(k_nn)+'_m'+str(m)+'_rep'+str(i)
            check_folder(folder_out)
            print(folder_out)
            filelog = folder_out+'/params.txt'
            with open(filelog,'w+') as log:
                log.write(params.replace('<folder_out>',folder_out).replace('<k_nn>',str(k_nn)).replace('<m>',str(m)).replace('<seed>',str(seed)))
            test_kpnn_init(folder_out, rho, eps_delta, arr_dict_traj[j], p, k_nn, gamm_fact=-2.0)
    

def test_kpnn_m2(folder_in, num_rep, ms, ks, rho, eps_delta, seeds, run_range, folder_out_prefix='./results/',delta=1e-10, visit_txt='', b_cont=False, b_mp=False,R=6371000.0):
    if visit_txt != '':
        dict_visit = get_visited_squares(visit_txt,R=R,suff='')
    else:
        dict_visit = get_visited_squares(folder_in,R=R)
    pois = list(dict_visit.keys())
    dict_traj = Extract_Cab_Data_All(folder_in,R=R)
    num_ks = len(ks) 
    params = 'foder_in;folder_out;m_len;k_nn;rho;delta;eps_delta;seed;visit_txt\n'
    params = params + folder_in+';<folder_out>;<m>;<k_nn>;'+str(rho)+';'+str(delta)+';'+str(eps_delta)+';<seed>;'+visit_txt
    if not(b_mp):
        for i in run_range:
            arr_dict_traj = [{} for m in ms]
            np.random.seed(seeds[i])
            poi = np.random.choice(pois, 1)
            p = convert_visit_key(poi[0])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                for j in range(len(ms)):
                    m_len = ms[j]
                    indices = get_m_indices(n, m_len, b_cont=b_cont)
                    arr_dict_traj[j][key] = x[indices]
            test_kpnn_m_single2(folder_out_prefix, ms, ks, rho, eps_delta, arr_dict_traj, p, seeds[num_rep+i], params, i)
    else:
        procs = []
        for i in run_range:
            arr_dict_traj = [{} for m in ms]
            np.random.seed(seeds[i])
            poi = np.random.choice(pois, 1)
            p = convert_visit_key(poi[0])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                for j in range(len(ms)):
                    m_len = ms[j]
                    indices = get_m_indices(n, m_len, b_cont=b_cont)
                    arr_dict_traj[j][key] = x[indices]
            procs.append(Process(target=test_kpnn_m_single2,args=[folder_out_prefix, ms, ks, rho, eps_delta, arr_dict_traj, p, seeds[num_rep+i], params, i]))
            procs[i-run_range[0]].start()
        for i in run_range:
            procs[i-run_range[0]].join()


def test_kpnn_rho_single2(folder_out_prefix, m_len, ks, rhos, eps_deltas, dict_traj, p, seed, params, i):
    num_rhos = len(rhos)
    num_ks = len(ks)
    np.random.seed(seed)
    for k in range(num_ks):
        k_nn = ks[k]
        if k_nn >= 125:
            continue
        for j in range(num_rhos):
            rho = rhos[j]
            eps_delta = eps_deltas[j]        
            folder_out = folder_out_prefix+'test_kpnn2_rho'+str(rho)+'_k'+str(k_nn)+'_m'+str(m_len)+'_rep'+str(i)
            check_folder(folder_out)
            print(folder_out)
            filelog = folder_out+'/params.txt'
            with open(filelog,'w+') as log:
                log.write(params.replace('<folder_out>',folder_out).replace('<k_nn>',str(k_nn)).replace('<rho>',str(rho)).replace('<eps_delta>',str(eps_delta)).replace('<seed>',str(seed)))
            test_kpnn_init(folder_out, rho, eps_delta, dict_traj, p, k_nn, gamm_fact=-2.0)


def test_kpnn_rho2(folder_in, num_rep, m_len, ks, rhos, eps_deltas, seeds, run_range, folder_out_prefix='./results/',delta=1e-10, visit_txt='', b_cont=False, b_mp=False, R=6371000.0):
    if visit_txt != '':
        dict_visit = get_visited_squares(visit_txt,R=R,suff='')
    else:
        dict_visit = get_visited_squares(folder_in,R=R)
    pois = list(dict_visit.keys())
    dict_traj = Extract_Cab_Data_All(folder_in,R=R)
    num_ks = len(ks)
    params = 'foder_in;folder_out;m_len;k_nn;rho;delta;eps_delta;seed;visit_txt\n'
    
    params = params + folder_in+';<folder_out>;'+str(m_len)+';<k_nn>;<rho>;'+str(delta)+';<eps_delta>;<seed>;'+visit_txt
    if not(b_mp):
        for i in run_range:
            dict_traj_i = {}
            np.random.seed(seeds[i])
            poi = np.random.choice(pois, 1)
            p = convert_visit_key(poi[0])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                indices = get_m_indices(n, m_len, b_cont=b_cont)
                dict_traj_i[key] = x[indices]
            test_kpnn_rho_single2(folder_out_prefix, m_len, ks, rhos, eps_deltas, dict_traj_i, p, seeds[num_rep+i], params, i)
    else:
        procs = []
        for i in run_range:
            dict_traj_i = {}
            np.random.seed(seeds[i])
            poi = np.random.choice(pois, 1)
            p = convert_visit_key(poi[0])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                indices = get_m_indices(n, m_len, b_cont=b_cont)
                dict_traj_i[key] = x[indices]
            procs.append(Process(target=test_kpnn_rho_single2,args=[folder_out_prefix, m_len, ks, rhos, eps_deltas, dict_traj_i, p, seeds[num_rep+i], params, i]))
            procs[i-run_range[0]].start()
        for i in run_range:
            procs[i-run_range[0]].join()


def test_convh_m_single(folder_out_prefix, ms, rho, eps_delta, arr_dict_traj, seed, params, i):
    num_ms = len(ms)
    np.random.seed(seed)
    for j in range(num_ms):
        m = ms[j]    
        folder_out = folder_out_prefix+'test_convh_rho'+str(rho)+'_m'+str(m)+'_rep'+str(i)
        check_folder(folder_out)
        print(folder_out)
        filelog = folder_out+'/params.txt'
        with open(filelog,'w+') as log:
            log.write(params.replace('<folder_out>',folder_out).replace('<m>',str(m)).replace('<seed>',str(seed)))
        test_convh_init(folder_out, rho, eps_delta, arr_dict_traj[j])


def test_convh_m(folder_in, num_rep, ms, rho, eps_delta, seeds, run_range, folder_out_prefix='./results/', delta=1e-10, b_cont=False, b_mp=False, R=6371000.0):
    dict_traj = Extract_Cab_Data_All(folder_in,R=R)
    params = 'foder_in;folder_out;m_len;rho;delta;eps_delta;seed\n'
    params = params + folder_in+';<folder_out>;<m>;'+str(rho)+';'+str(delta)+';'+str(eps_delta)+';<seed>'
    if not(b_mp):
        for i in run_range:
            arr_dict_traj = [{} for m in ms]
            np.random.seed(seeds[i])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                for j in range(len(ms)):
                    m_len = ms[j]
                    indices = get_m_indices(n, m_len, b_cont=b_cont)
                    arr_dict_traj[j][key] = x[indices]
            test_convh_m_single(folder_out_prefix, ms, rho, eps_delta, arr_dict_traj, seeds[num_rep+i], params, i)
    else:
        procs = []
        for i in run_range:
            arr_dict_traj = [{} for m in ms]
            np.random.seed(seeds[i])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                for j in range(len(ms)):
                    m_len = ms[j]
                    indices = get_m_indices(n, m_len, b_cont=b_cont)
                    arr_dict_traj[j][key] = x[indices]
            procs.append(Process(target=test_convh_m_single,args=[folder_out_prefix, ms, rho, eps_delta, arr_dict_traj, seeds[num_rep+i], params, i]))
            procs[i-run_range[0]].start()
        for i in run_range:
            procs[i-run_range[0]].join()


def test_convh_rho_single(folder_out_prefix, m_len, rhos, eps_deltas, dict_traj, seed, params, i):
    num_rhos = len(rhos)
    np.random.seed(seed)
    for j in range(num_rhos):
        rho = rhos[j]
        eps_delta = eps_deltas[j]        
        folder_out = folder_out_prefix+'test_convh_rho'+str(rho)+'_m'+str(m_len)+'_rep'+str(i)
        check_folder(folder_out)
        print(folder_out)
        filelog = folder_out+'/params.txt'
        with open(filelog,'w+') as log:
            log.write(params.replace('<folder_out>',folder_out).replace('<rho>',str(rho)).replace('<eps_delta>',str(eps_delta)).replace('<seed>',str(seed)))
        test_convh_init(folder_out, rho, eps_delta, dict_traj)


def test_convh_rho(folder_in, num_rep, m_len, rhos, eps_deltas, seeds, run_range, folder_out_prefix='./results/', delta=1e-10, b_cont=False, b_mp=False, R=6371000.0):
    dict_traj = Extract_Cab_Data_All(folder_in,R=R)
    params = 'foder_in;folder_out;m_len;rho;delta;eps_delta;seed\n'
    params = params + folder_in+';<folder_out>;'+str(m_len)+';<rho>;'+str(delta)+';<eps_delta>;<seed>'
    if not(b_mp):
        for i in run_range:
            dict_traj_i = {}
            np.random.seed(seeds[i])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                indices = get_m_indices(n, m_len, b_cont=b_cont)
                dict_traj_i[key] = x[indices]
            test_convh_rho_single(folder_out_prefix, m_len, rhos, eps_deltas, dict_traj_i, seeds[num_rep+i], params, i)
    else:
        procs = []
        for i in run_range:
            dict_traj_i = {}
            np.random.seed(seeds[i])
            traj_keys = np.random.choice(list(dict_traj.keys()), 50)
            for key in traj_keys:
                x = dict_traj[key]
                n = len(x)
                indices = get_m_indices(n, m_len, b_cont=b_cont)
                dict_traj_i[key] = x[indices]
            procs.append(Process(target=test_convh_rho_single,args=[folder_out_prefix, m_len, rhos, eps_deltas, dict_traj_i, seeds[num_rep+i], params, i]))
            procs[i-run_range[0]].start()
        for i in run_range:
            procs[i-run_range[0]].join()

