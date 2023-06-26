"""
scan sry file with

find  . -name *.sry > /sps/grand/colley/list_sry.txt &

@author: jcolley
"""

import os.path
import re
import sqlite3
from sqlite3 import Error

import numpy as np
import matplotlib.pyplot as plt

from sradio.io.shower.zhaires_txt import ZhairesSingleEventText

root_path = "/sps/grand/tueros"
# path_scan = "/home/jcolley/projet/grand_wk/data/zhaires/list_sry.txt"
path_scan = "/sps/grand/colley/db/list_sry.txt"
path_scan = "/home/jcolley/projet/grand_wk/data/zhaires/list_sry.txt"


# approximative regular expression of string float
REAL = "[+-]?[0-9][0-9.eE+-]*"
L_primary = ["Proton", "Gamma", "Iron"]


def create_sqlite_db(path_db):
    """create a database connection to a SQLite database"""
    conn = sqlite3.connect(path_db)
    print(sqlite3.version)
    return conn


def fill_table_shower_pars(con_db, path_simu, shower_pars):
    # Creating a cursor object using the cursor() method
    cursor = con_db.cursor()

    # Doping table if already exists.
    cursor.execute("DROP TABLE IF EXISTS Shower")

    # Creating table as per requirement
    # altitude UNSIGNED TINYINT
    sql = """CREATE TABLE Shower(
       path TEXT,
       prim_part TEXT,
       energy REAL,
       elevation REAL,
       azimuth REAL,
       altitude REAL
    )"""
    cursor.execute(sql)

    # Commit your changes in the database
    con_db.commit()

    # fill data
    for idx, c_path in enumerate(path_simu):
        pars = shower_pars[idx]
        type_p = str(pars[0], "UTF-8")
        data = rf"('{c_path}','{type_p}',{pars[1]},{pars[2]},{pars[3]},{pars[4]})"
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
    sql = """CREATE TABLE PathRoot(
       path TEXT
    )"""
    cursor.execute(sql)

    # Commit your changes in the database
    con_db.commit()
    cursor.execute(f"INSERT INTO PathRoot VALUES ('{root_simu}')")
    con_db.commit()


def parser_scan_name(path_scan):
    '''
    Return list on path simulation and numpy structured array of parameters simulation
    
    :param path_scan:
    '''
    l_path = []
    a_dtype = {
        "names": ("primary", "energy", "elevation", "azimuth", "dist"),
        "formats": ("S20", "f4", "f4", "f4", "uint8"),
    }

    with open(path_scan) as f_scan:
        l_path_sry = f_scan.readlines()
    f_err = open("error.txt", "w")
    nb_sim = len(l_path_sry)
    pars_sim = np.zeros(nb_sim, dtype=a_dtype)
    print(pars_sim)
    idx_ok = 0
    idx_sry_ok = 0
    for idx, p_sry in enumerate(l_path_sry):
        # if idx == 10: break
        s_path = p_sry.split("/")
        name_sry = s_path[-1]
        s_elt = name_sry.split("_")
        primary = ""
        for elt in s_elt:
            if elt in L_primary:
                primary = elt
        if primary == "":
            # TODO : read sry file to extract AuthorityInformationAccess
            idx_sry = p_sry.find(name_sry)
            abs_dir = os.path.join(root_path, p_sry[2:idx_sry])
            print(f"Read  {abs_dir}")
            zh_txt = ZhairesSingleEventText(abs_dir)
            if not zh_txt.read_summary_file():
                f_err.write(f"\n{abs_dir} nok read")
                continue
            print(zh_txt.d_info)
            unit = zh_txt.d_info["energy"]["unit"]
            if unit == "PeV":
                energy = 1e-3 * zh_txt.d_info["energy"]["value"]
            elif unit == "EeV":
                energy = zh_txt.d_info["energy"]["value"]
            else:
                f_err.write(f"\n{abs_dir} pb unit {unit}")
                energy = -2
            convert = (
                zh_txt.d_info["primary"],
                energy,
                zh_txt.d_info["shower_zenith"],
                (zh_txt.d_info["shower_azimuth"] % 360),
                zh_txt.d_info["x_max"]["dist"],
            )
            idx_sry_ok += 1
            pars_sim[idx_ok] = convert
        else:
            f_re = rf"\w+_(?P<energy>{REAL})_(?P<elevation>{REAL})_(?P<azimuth>{REAL})"
            ret = re.search(f_re, name_sry)
            if not isinstance(ret, re.Match):
                # TODO : read sry file to extract information
                # print(f"Can't find parameters in: {p_sry} with re: {f_re}")
                f_err.write(f"\n{abs_dir} re file NOK")
                continue
            d_pars = ret.groupdict()
            try:
                convert = (
                    primary,
                    float(d_pars["energy"]),
                    float(d_pars["elevation"]),
                    float(d_pars["azimuth"]),
                    0,
                )
                pars_sim[idx_ok] = convert
            except:
                f_err.write(f"\n{p_sry}: to float nok {d_pars}")
                continue
        idx_ok += 1
        # path simu
        idx_f = p_sry.find(name_sry)
        l_path.append(p_sry[2 : idx_f - 1])
    print(pars_sim)
    pars_sim = pars_sim[:idx_ok]
    print(f"{nb_sim-idx_ok} convert failed on {nb_sim}")
    print(pars_sim[:-5])
    print(l_path[:5])
    assert pars_sim.shape[0] == len(l_path)
    #
    f_err.close()
    return pars_sim, l_path


def zhaires_stat(path_scan):
    pars_sim, l_path = parser_scan_name(path_scan)
    plt.figure()
    plt.title("Energy")
    plt.hist(pars_sim["energy"], log=True)
    plt.xlabel("EeV")
    plt.figure()
    plt.title("Shower direction: elevation")
    plt.hist(pars_sim["elevation"], log=True)
    plt.xlabel("deg")
    plt.figure()
    plt.title("Shower direction: azimuth")
    plt.hist(pars_sim["azimuth"], log=True)
    plt.xlabel("deg")


def zhaires_master_create(path_scan, name_db, root_scan):
    pars, l_path = parser_scan_name(path_scan)
    con_db = create_sqlite_db(name_db)
    fill_table_shower_pars(con_db, l_path, pars)
    fill_table_path(con_db, root_scan)
    con_db.close()


if __name__ == "__main__":
    zhaires_stat(path_scan)
    # zhaires_master_create(path_scan, "zhaires_tueros3.db", "/sps/grand/tueros")
    plt.show()
