#!/usr/bin/env python3

import sys
import argparse
import logging
import multiprocessing

import game_master
import pokemon

"""Precompute a JSON table of optimal PvP IVs."""


class IVOptimizer():
    full_precision = True
    simd = True

    def __init__(self, gm, full_precision=True, simd=True):
        self.gm = gm
        self.full_precision = full_precision
        self.simd = simd

    def find_optimal_proc(self, names):
        """Worker process to compute optimal IVs for 1500 and 2500 cp"""
        optimal = {}
        p = pokemon.Pokemon(self.gm)
        for name in names:
            p.update(name=name, fast=pokemon.VAL.DONT_SET, charged=pokemon.VAL.DONT_SET)
            assert pokemon.Pokemon.iv_cache is not None
            for cp_max in [500, 1500, 2500, None]:
                o = None
                for level_max in [self.gm.K_LEVEL_MAX, 40]:
                    if o is None or o[3] > level_max:
                        o = p.optimize_iv(cp_max,
                                          level_max,
                                          full_precision=self.full_precision,
                                          simd=self.simd)
                    # make sure we didn't exceed level_max
                    assert(o[3] <= level_max)
                    if o:
                        optimal.setdefault(name, {}).setdefault(level_max, {})[cp_max] = o
        return optimal


def chunk(l, n):
    """Divide a list into n-sized chunks."""
    n = max(1, n)
    for i in range(0, len(l), n):
        yield l[i: i + n]


def find_optimal_multi(gm, args, full_precision=True, simd=True):
    """Coordinating process to run find_optimal_proc across multiple procs"""
    optimizer = IVOptimizer(gm, full_precision=full_precision, simd=simd)
    iv_cache = pokemon.Cache(args.output)
    # forms to optimize
    if args.forms:
        missing = args.forms
    else:
        missing = gm.pokemon.keys()
    # Find which pokemon are not in the list
    missing = [x for x in missing if x not in iv_cache]
    if args.npokemon is not None:
        missing = missing[:args.npokemon]
    threads = args.threads[0]
    print("Computing optimal IVs for", len(missing), "pokemon using", threads, threads > 1 and "threads" or "thread")
    try:
        if threads == 1:
            # just run serially
            od = optimizer.find_optimal_proc(missing)
            iv_cache.update(od)
        else:
            missing = chunk(missing, 20)
            with multiprocessing.Pool(threads) as pool:
                for rv in pool.imap_unordered(optimizer.find_optimal_proc, missing):
                    iv_cache.update(rv)
                    sys.stdout.write(".")
                    sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    sys.stdout.write("\n")
    iv_cache.write()
    return iv_cache


if __name__ == "__main__":
    """Not intended for standalone use."""
    parser = argparse.ArgumentParser(description='Retrieve and parse a GAME_MASTER.json file')
    parser.add_argument("-v", dest="verbose", help="verbose output", action="store_true")
    parser.add_argument("-t", dest="threads", help="threads", nargs=1, type=int, default=[multiprocessing.cpu_count()])
    parser.add_argument("-n", dest="npokemon", help="number of pokemon to optimize", type=int)
    parser.add_argument("-o", dest="output", help="output JSON file", default=pokemon.OPTIMAL_IV)
    parser.add_argument("--serial", help="evaluate serially", action="store_true")
    parser.add_argument("--low-precision", dest="low_precision", help="calculate CP and stat product per published tables", action="store_true")
    parser.add_argument("forms", nargs="*", help="forms to evaluate")
    args = parser.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
        log_format = "%(levelname)-5s %(funcName)s:%(lineno)d > %(message)s\n"
    else:
        log_level = logging.INFO
        log_format = "%(message)s"

    logging.basicConfig(level=log_level, format=log_format)

    simd = not args.serial
    full_precision = not args.low_precision

    gm = game_master.GameMaster()
    optimal = find_optimal_multi(gm, args, full_precision=full_precision, simd=simd)

    # reorder by highest stat product
    best = {}
    for name, cps in optimal.items():
        cps = cps[gm.K_LEVEL_MAX]
        for cp, data in cps.items():
            best.setdefault(cp, {})[data[5]] = list(data[:5]) + [name]
    for cp, monsters in best.items():
        print()
        print("CP:", cp)
        for sp in sorted(monsters.keys())[-5:]:
            print(sp, monsters[sp])
