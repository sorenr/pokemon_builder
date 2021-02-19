import sys
import math
import game_master
import numpy as np

"""
Find "fraternal twins": forms with the same base stats but different types or attacks.
"""

DTYPE = np.uint32


def get_ads(gm, name):
    stats = gm.pokemon[name]['stats']
    ads = [stats[gm.K_BASE_ATTACK], stats[gm.K_BASE_DEFENSE], stats[gm.K_BASE_STAMINA]]
    return np.array(ads)


def find_closest(target):
    gm = game_master.GameMaster()

    target_ads = get_ads(gm, target)

    mons = {}

    for name, val in gm.pokemon.items():
        if name == target:
            continue
        ads = get_ads(gm, name)
        ads -= target_ads
        print(ads)
        stats = np.sum(np.power(ads, 2))
        stats = math.sqrt(stats)
        print(stats)
        grp = mons.setdefault(stats, set())
        add = True
        for n in grp:
            # remove mons starting with this prefix
            if n.startswith(name):
                grp.remove(n)
            # skip this one if it's a prefix
            if name.startswith(n):
                add = False
                break
        if add:
            grp.add(name)

    for key, mset in sorted(mons.items()):
        if len(mset) > 1:
            print("{0:0.2f}".format(key), list(sorted(mset)))


if __name__ == "__main__":
    find_closest(sys.argv[1])
