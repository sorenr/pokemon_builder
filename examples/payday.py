import game_master
import pprint
import sys

BASE_DUST = 100

def dedup(gm, names):
    rv = set()
    for name in sorted(names):
        dust = gm.pokemon[name]['encounter'].get('bonusStardustCaptureReward')
        id = gm.pokemon[name]['pokemonId']
        id_dust = gm.pokemon[id]['encounter'].get('bonusStardustCaptureReward')
        if dust == id_dust:
            assert name.startswith(id)
            rv.add(id)
        else:
            print("VARIANT BONUS", name)
            rv.add(name)
    return sorted(list(rv))

gm = game_master.GameMaster()

bonuses = {}

for name, stats in gm.pokemon.items():
    bonus = stats['encounter'].get('bonusStardustCaptureReward')
    if bonus is None:
        continue
    bonuses.setdefault(BASE_DUST + bonus, []).append(name)

for bonus, names in bonuses.items():
    bonuses[bonus] = dedup(gm, names)

for bonus, names in sorted(bonuses.items(), reverse=True):
    num_names = len(names)
    names = ", ".join(names)
    print(f'{bonus} ({num_names}) {names}')
    print()
