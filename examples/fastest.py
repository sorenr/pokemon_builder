import game_master
import sys
import math

gm = game_master.GameMaster()

results = {}

moves = gm.moves_combat

for name in sorted(gm.pokemon):
    if name.startswith("SMEARGLE"):
        continue
    for combo in sorted(gm.move_combinations(name, False, False, r=1)):
        fast = combo[0]
        charged = combo[1][0]
        try:
            fe = moves[fast]['energyDelta']
            ft = moves[fast].get('durationTurns', 0) + 1
            ce = -moves[charged]['energyDelta']
        except KeyError:
            continue

        t = math.ceil(ce / fe) * ft
        results.setdefault(t, []).append([name, fast, charged])

for k, v in sorted(results.items()):
    print(k)
    print(v)
    print()
