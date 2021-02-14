import game_master

"""
Find "fraternal twins": forms with the same base stats but different types or attacks.
"""


def find_twins():
    gm = game_master.GameMaster()

    mons = {}

    for name, val in gm.pokemon.items():
        stats = val['stats']
        stats = (stats[gm.K_BASE_ATTACK], stats[gm.K_BASE_DEFENSE], stats[gm.K_BASE_STAMINA])
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

    for combo, mset in sorted(mons.items()):
        if len(mset) > 1:
            print(list(sorted(mset)))


if __name__ == "__main__":
    find_twins()
