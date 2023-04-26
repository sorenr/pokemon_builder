#!/usr/bin/env python3

import game_master
import pprint
import numpy
import matplotlib
import pandas
import seaborn
import datetime


def GetStats(gm: game_master.GameMaster, name: str):
    data = gm.pokemon[name]
    stats = data[game_master.GameMaster.K_STATS]
    a = stats[game_master.GameMaster.K_BASE_ATTACK]
    d = stats[game_master.GameMaster.K_BASE_DEFENSE]
    s = stats[game_master.GameMaster.K_BASE_STAMINA]
    family = data[game_master.GameMaster.K_ID_UNIQUE]
    return (family, a, d, s)


def GetType(gm: game_master.GameMaster, name: str):
    data = gm.pokemon[name]
    t1 = data[game_master.GameMaster.K_TYPE1]
    t2 = data.get(game_master.GameMaster.K_TYPE2, t1)
    p_type = tuple(sorted(game_master.Types[x] for x in {t1, t2}))
    return p_type


def GetData():
    gm = game_master.GameMaster()
    print()

    names = sorted(gm.pokemon.keys())

    print(names)

    p_types = {}

    for name in names:
        p_type = GetType(gm, name)
        p_stats = GetStats(gm, name)
        p_types.setdefault(p_type, set()).add(p_stats)

    return p_types


def PlotData(data:dict):
    pprint.pprint(data)

    grid_np = numpy.zeros( (18, 18), dtype=int )
    for k, v in data.items():
        i = k[0].value
        if len(k) == 1:
            j = i
        else:
            j = k[1].value
        if i < j:
            i, j = j, i
        grid_np[i,j] = len(v)

    labels = [str(lbl).lower() for lbl in game_master.Types]

    print(grid_np)

    grid_pd = pandas.DataFrame(grid_np, index=labels, columns=labels)

    norm = matplotlib.colors.LogNorm()

    ax = seaborn.heatmap(grid_pd, annot=True, norm=norm)
    dt = datetime.datetime.now()
    ax.set_title("Pokémon Go Type Census {0}".format(dt.strftime("%m/%Y")))
    ax.text(18, 22, "u/bramlet", fontsize=9)
    fig = ax.get_figure()

    # cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='YlGnBu'), format="%0.1f", drawedges=True)

    fig.savefig("output.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    numpy.random.seed(0)
    seaborn.set_theme()
    data = GetData()
    PlotData(data)
