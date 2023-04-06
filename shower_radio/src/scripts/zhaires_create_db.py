'''
scan sry file with

find  . -name *.sry > /sps/grand/colley/list_sry.txt &

@author: jcolley
'''

import re
import sqlite3
from sqlite3 import Error

import numpy as np

path_scan = "/home/jcolley/projet/grand_wk/data/zhaires/list_sry.txt"
L_primary = ["Proton", "Gamma", "Iron"]
# approximative regular expression of string float
REAL = "[+-]?[0-9][0-9.eE+-]*"


def create_sqlite_db(path_db):
    """ create a database connection to a SQLite database """
    conn = sqlite3.connect(path_db)
    print(sqlite3.version)    
    return conn


def fill_table_shower_pars(con_db, path_simu, shower_pars):
    # Creating a cursor object using the cursor() method
    cursor = con_db.cursor()
    
    # Doping table if already exists.
    cursor.execute("DROP TABLE IF EXISTS Shower")
    
    # Creating table as per requirement
    sql = '''CREATE TABLE Shower(
       path TEXT,
       prim_part TEXT,
       energy REAL,
       dist_zen REAL,
       azimuth REAL
    )'''
    cursor.execute(sql)
    
    # Commit your changes in the database
    con_db.commit()
    
    # fill data
    for idx, c_path in enumerate(path_simu):
        pars = shower_pars[idx]
        type_p = str(pars[0], 'UTF-8')
        data = fr"('{c_path}','{type_p}',{pars[1]},{pars[2]},{pars[3]})"
        if idx % 100 == 0:
            print(idx)
        cursor.execute(f"INSERT INTO Shower VALUES {data}")
    con_db.commit()

def fill_table_path(con_db, root_simu):
    # Creating a cursor object using the cursor() method
    cursor = con_db.cursor()
    
    # Doping table if already exists.
    cursor.execute("DROP TABLE IF EXISTS PathRoot")
    
    # Creating table as per requirement
    sql = '''CREATE TABLE PathRoot(
       path TEXT
    )'''
    cursor.execute(sql)
    
    # Commit your changes in the database
    con_db.commit()
    cursor.execute(f"INSERT INTO PathRoot VALUES ('{root_simu}')")
    con_db.commit()

def parser_scan_name(path_scan):
    l_path = []
    a_dtype = {
            "names": ("primary", "energy", "dist_zen", "azimuth"),
            "formats": ("S20", "f4", "f4", "f4"),
        }

    with open(path_scan) as f_scan:
        l_path_sry = f_scan.readlines()
        
    nb_sim = len(l_path_sry)
    pars_sim = np.zeros(nb_sim, dtype=a_dtype)
    print(pars_sim)
    idx_ok = 0
    for idx, p_sry in enumerate(l_path_sry):
        #if idx == 10: break
        s_path = p_sry.split('/')
        name_sry = s_path[-1]
        s_elt = name_sry.split('_')
        primary = ""
        for elt in s_elt:
            if elt in L_primary:
                primary = elt
        # print(primary)
        f_re = fr"\w+_(?P<energy>{REAL})_(?P<dist_zen>{REAL})_(?P<azimuth>{REAL})"
        ret = re.search(f_re, name_sry)
        if not isinstance(ret, re.Match):
            print(f"Can't find parameters in: {p_sry} with re: {f_re}")
            continue
        d_pars = ret.groupdict()
        try:
            convert = (primary, float(d_pars["energy"]), float(d_pars["dist_zen"]), float(d_pars["azimuth"]))
            pars_sim[idx_ok] = convert
            idx_ok += 1
        except:
            print(f"Can't convert {convert}")
            continue
        # path simu
        idx_f = p_sry.find(name_sry)
        l_path.append(p_sry[2:idx_f - 1])
        print(l_path[-1])
    print(pars_sim)
    pars_sim = pars_sim[:idx_ok]
    print(f"{nb_sim-idx_ok} convert failed on {nb_sim}")
    print(pars_sim[:-5])
    print(l_path[:5])
    assert pars_sim.shape[0] == len(l_path)
    return pars_sim, l_path


        

if __name__ == '__main__':
    pars, l_path = parser_scan_name(path_scan)
    con_db = create_sqlite_db("zhaires_tueros2.db")
    fill_table_shower_pars(con_db, l_path, pars)
    fill_table_path(con_db, "/sps/grand/tueros")
    con_db.close()