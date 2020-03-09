#!/usr/bin/env python3

import sys
import multiprocessing

import game_master
import pokemon


"""Precompute a JSON table of optimal PvP IVs."""


# default path for the optimal PVP table
OPTIMAL_IV = "OPTIMAL_IV.json"


def find_optimal_proc(name):
    """Worker process to compute optimal IVs for 1500 and 2500 cp"""
    p = pokemon.Pokemon(GM, name=name)
    optimal = {}
    for mcp in [1500, 2500, None]:
        o = p.optimize_iv(max_cp=mcp)
        if o:
            optimal.setdefault(name, {})[mcp] = o
    return optimal


def find_optimal_multi(out=OPTIMAL_IV):
    """Coordinating process to run find_optimal_proc across multiple procs"""
    optimal = pokemon.Cache(out)
    # Find which pokemon are not in the list
    missing = [x for x in GM.pokemon.keys() if x not in optimal]
    print("Computing optimal IVs for", len(missing), "pokemon")
    with multiprocessing.Pool() as pool:
        try:
            for rv in pool.imap_unordered(find_optimal_proc, missing):
                optimal.update(rv)
                sys.stdout.write(".")
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass
    sys.stdout.write("\n")
    optimal.write()
    return optimal


if __name__ == "__main__":
    GM = game_master.GameMaster()
    if len(sys.argv) > 1:
        optimal = find_optimal_multi(sys.argv[1])
    else:
        optimal = find_optimal_multi()

    # reorder by highest stat product
    best = {}
    for name, cps in optimal.items():
        for cp, data in cps.items():
            best.setdefault(cp, {})[data[5]] = list(data[:5]) + [name]
    for cp, monsters in best.items():
        print()
        print("CP:", cp)
        for sp in sorted(monsters.keys())[-5:]:
            print(int(sp), monsters[sp])
