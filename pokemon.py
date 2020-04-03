#!/usr/bin/env python3

import logging
import argparse
import random
import unittest
import enum
import time
import json
import os
import fnmatch

import game_master


# default path for the optimal PVP table
OPTIMAL_IV = "OPTIMAL_IV.json"


class VAL(enum.Enum):
    """Actions for setting pokemon attributes."""
    DONT_SET = enum.auto()  # don't do anything
    RANDOM = enum.auto()    # set a random value
    OPTIMAL = enum.auto()   # set an optimal value


class CONTEXT(enum.Enum):
    """Contexts for evaluating an interaction"""
    BATTLE = enum.auto()  # raid
    COMBAT = enum.auto()  # pvp


def clamp(val, vmin, vmax):
    """Clamp val between vmin and vmax"""
    return max(vmin, min(vmax, val))


class Move():
    """Attack move representation"""

    _FSUFFIX = "_FAST"
    _FL = len(_FSUFFIX)

    def __init__(self, pokemon, name):
        self.pokemon = pokemon
        self.gm = self.pokemon.gm
        self.update(name)

    def update(self, name):
        """Set the attack name and the data depending on that name."""
        self.name = name
        try:
            self.data = self.pokemon.moves[self.name]
        except KeyError:
            raise game_master.PokemonKeyError(self.name, self.pokemon.moves)
        self.buffs = self.data.get("buffs")
        self.cooldown = self.data.get('durationTurns', 0)
        self.energy_delta = self.data.get('energyDelta', 0)
        self.type = game_master.Types[self.data['type']]

        # set attack bonus based on fast/charged
        self.is_fast = self.name.endswith(Move._FSUFFIX)
        if self.is_fast:
            self.bonus_atk = self.pokemon.settings.get('fastAttackBonusMultiplier', 1)
        else:
            self.bonus_atk = self.pokemon.settings.get('chargeAttackBonusMultiplier', 1)

        legacy = game_master.GameMaster._LEGACY.get(self.name, [])
        for pattern in legacy:
            if fnmatch.fnmatch(self.pokemon.name, pattern):
                self.is_legacy = True
                break
        else:
            self.is_legacy = False

        self.power = self.data.get('power', 0)
        if self.type in self.pokemon.type:
            self.stab = self.pokemon.settings['sameTypeAttackBonusMultiplier']
        else:
            self.stab = 1

    def damage(self, target):
        """Calculate damage dealt to the target."""
        if not self.power:
            return 1
        attack = self.pokemon.attack * self.bonus_atk
        if self.pokemon.is_purified and target is not None and target.is_shadow:
            attack *= self.pokemon.settings.get('purifiedPokemonAttackMultiplierVsShadow', 1)
        attack *= self.pokemon.gm.buff_multiplier_attack[self.pokemon.buff_attack]

        defense = 1
        effective = 1
        if target is not None:
            defense *= target.defense * target.bonus_def
            defense *= target.gm.buff_multiplier_defense[target.buff_defense]
            effective *= self.gm.effect(self.type, target.type)

        return 1 + int(0.5 * self.power * attack / defense * self.stab * effective)

    def attack(self, target, damage=None, shield=True):
        """Attack the target, compute results including buffs."""
        # record the target's old stats
        hp_old = target.hp
        energy_old = self.pokemon.energy

        # compute damage if it wasn't precomputed
        if damage is None:
            damage = self.damage(target)

        # if it's a charged attack
        if not self.is_fast:
            # damage is 1 when shielding
            if shield and target.shields > 0:
                target.shields -= 1
                damage = 1
            target.cooldown += 2

        # apply the attack damage
        target.hp -= damage
        self.pokemon.energy += self.energy_delta
        self.pokemon.cooldown -= self.cooldown
        logging.debug("%s-%s->%s dmg=%d hp=%0.1f->%0.1f en=%0.1f->%0.1f",
                      self.pokemon.name, self.name, target.name, damage,
                      hp_old, target.hp, energy_old, self.pokemon.energy)

        # apply attack buffs
        # FIXME: add expected value for lower-probability buffs
        if self.buffs and self.buffs['buffActivationChance'] >= 1.0:
            self.pokemon.buff_attack += self.buffs.get('attackerAttackStatStageChange', 0)
            self.pokemon.buff_attack = clamp(self.pokemon.buff_attack, 0, len(self.pokemon.gm.buff_multiplier_attack) - 1)
            self.pokemon.buff_defense += self.buffs.get('attackerDefenseStatStageChange', 0)
            self.pokemon.buff_defense = clamp(self.pokemon.buff_defense, 0, len(self.pokemon.gm.buff_multiplier_defense) - 1)
            target.buff_attack += self.buffs.get('targetAttackStatStageChange', 0)
            target.buff_attack = clamp(target.buff_attack, 0, len(target.gm.buff_multiplier_attack) - 1)
            target.buff_defense += self.buffs.get('targetDefenseStatStageChange', 0)
            target.buff_defense = clamp(target.buff_defense, 0, len(target.gm.buff_multiplier_defense) - 1)

    def __str__(self):
        """Return the attack name, minus the suffix."""
        suffix = self.is_legacy and '*' or ''
        if self.name.endswith(self._FSUFFIX):
            return self.name[:-self._FL] + suffix
        return self.name + suffix


class Cache(dict):
    """A dict which persists."""
    def __init__(self, path):
        self.path = path
        try:
            with open(path) as fd:
                self.update(json.loads(fd.read()))
            logging.debug("Loaded %d entries from %s", len(self), self.path)
        except FileNotFoundError:
            pass

    def write(self, newpath=None):
        """Write the cache before we exit."""
        if not newpath:
            newpath = self.path
        with open(newpath, 'w') as fd:
            json.dump(self, fd)
        logging.debug("Wrote %d entries to %s", len(self), self.path)
        return newpath


class Pokemon():
    """Full representation of a specific Pokemon: type, IVs, attacks."""

    fast = None
    charged = []
    iv_cache = None

    def __init__(self, gm,
                 name=VAL.RANDOM,
                 attack=VAL.RANDOM,
                 defense=VAL.RANDOM,
                 stamina=VAL.RANDOM,
                 level=VAL.RANDOM,
                 fast=VAL.RANDOM,
                 charged=VAL.RANDOM,
                 context=CONTEXT.COMBAT):
        self.gm = gm
        self.update(name=name, attack=attack, defense=defense,
                    stamina=stamina, level=level,
                    fast=fast, charged=charged,
                    context=context)

    def update(self,
               name=VAL.DONT_SET,
               attack=VAL.DONT_SET,
               defense=VAL.DONT_SET,
               stamina=VAL.DONT_SET,
               level=VAL.DONT_SET,
               fast=VAL.DONT_SET,
               charged=VAL.DONT_SET,
               context=VAL.DONT_SET):
        """Set or randomize values if requested."""

        # everything depends on context, so set context first
        if context is not VAL.DONT_SET:
            self.context = context
            if self.context == CONTEXT.COMBAT:
                self.moves = self.gm.moves_combat
                self.settings = self.gm.settings_combat
            elif self.context == CONTEXT.BATTLE:
                self.moves = self.gm.moves_battle
                self.settings = self.gm.settings_battle

        if name is VAL.RANDOM:
            name = random.choice(list(self.gm.pokemon.keys()))
        if name is not VAL.DONT_SET:
            self.name = name
            try:
                self.data = self.gm.pokemon[self.name]
            except KeyError:
                raise game_master.PokemonKeyError(self.name, self.gm.pokemon)
            self.stats = self.data['stats']
            self.type = {self.data['type1'], self.data.get('type2')}
            self.type = {game_master.Types[x] for x in self.type if x is not None}
            self.possible_fast = self.gm.possible_fast(name)
            self.possible_charged = self.gm.possible_charged(name)

            # we can't keep our moves if they're not legal moves
            if fast is VAL.DONT_SET and self.fast not in self.possible_fast:
                fast = VAL.RANDOM
            self.charged = [x for x in self.charged if x in self.possible_charged]
            if charged is VAL.DONT_SET and not self.charged:
                charged = VAL.RANDOM

        self.is_shadow = self.name.endswith(self.gm.K_SHADOW_SUFFIX)
        self.is_purified = self.name.endswith(self.gm.K_PURIFIED_SUFFIX)

        self.bonus_atk = 1
        self.bonus_def = self.settings.get(self.gm.K_BONUS_DEF, 1)

        if self.is_shadow:
            self.bonus_atk *= self.settings[self.gm.K_SHADOW_BONUS_ATK]
            self.bonus_def *= self.settings[self.gm.K_SHADOW_BONUS_DEF]

        if level is VAL.RANDOM:
            level = 0.5 * random.randint(2, 82)
        if level is not VAL.DONT_SET:
            self.level = level
            self.cpm = self.gm.cp_multiplier(self.level)

        if attack is VAL.RANDOM:
            attack = random.randint(0, 15)
        if attack is not VAL.DONT_SET:
            self.iv_attack = attack

        if defense is VAL.RANDOM:
            defense = random.randint(0, 15)
        if defense is not VAL.DONT_SET:
            self.iv_defense = defense

        if stamina is VAL.RANDOM:
            stamina = random.randint(0, 15)
        if stamina is not VAL.DONT_SET:
            self.iv_stamina = stamina

        self.attack = (self.iv_attack + self.stats['baseAttack']) * self.cpm
        self.defense = (self.iv_defense + self.stats['baseDefense']) * self.cpm
        self.stamina = (self.iv_stamina + self.stats['baseStamina']) * self.cpm

        self.defense *= self.settings.get(self.gm.K_BONUS_DEF, 1)

        # A move's stats depend on its pokemon, so set the pokemon stats before setting its moves.
        if fast is VAL.RANDOM:
            fast = random.choice(self.possible_fast)
        if fast is not VAL.DONT_SET:
            self.fast = isinstance(fast, Move) and fast or Move(self, fast)

        if charged is VAL.RANDOM:
            # pick 1 or 2 charged attacks
            p_charged = self.possible_charged.copy()
            charged = [random.choice(p_charged)]
            if len(p_charged) > 1 and random.random() > 0.5:
                p_charged.remove(charged[0])
                charged.append(random.choice(p_charged))
        if charged is not VAL.DONT_SET:
            # make charged Moves if it's not a Move already
            self.charged = [isinstance(c, Move) and c or Move(self, c) for c in charged]

        # reset to default combat values
        self.reset()

    def cp(self):
        # CP = (Attack * Defense^0.5 * Stamina^0.5 * CP_Multiplier^2) / 10
        # https://gamepress.gg/pokemongo/pokemon-stats-advanced#cp-multiplier
        rv = self.attack
        rv *= pow(self.defense, 0.5)
        rv *= pow(self.stamina, 0.5)
        return rv / 10

    def reset(self, shields=1):
        """Reset attack/defense/stamina after combat."""
        self.hp = self.stamina
        self.shields = shields
        self.buff_attack = 4
        self.buff_defense = 4
        self.cooldown = 0
        self.energy = 0

    def __str__(self):
        """Return a human-readable string representation of the pokemon."""
        iv_str = "{0:d}/{1:d}/{2:d}".format(self.iv_attack, self.iv_defense, self.iv_stamina)
        type_str = "/".join(sorted([str(x) for x in self.type]))
        moves_str = str(self.fast) + " " + "+".join(sorted([str(x) for x in self.charged]))
        cp = self.cp()
        return f"{self.name:s} ({type_str:s}) {iv_str:s} {self.level:0.1f} {moves_str:s} CP={cp:0.1f}"

    def __eq__(self, b):
        """Return whether we're comparing two identical pokemon."""
        return (self.name == b.name and
                self.iv_attack == b.iv_attack and
                self.iv_defense == b.iv_defense and
                self.iv_stamina == b.iv_stamina and
                self.level == b.level)

    def stat_product(self, full_precision=False):
        """Return the stat product, a useful metric for optimizing IVs."""
        # https://gostadium.club/pvp/iv
        # https://www.reddit.com/r/TheSilphRoad/comments/a4yb62/a_pvp_performance_calculator_featuring_calcy_iv
        sp = self.attack * self.defense
        if full_precision:
            return sp * self.stamina
        else:
            # most online stat tables publish this number
            sp *= int(self.stamina)
            return int(sp + 0.5)  # round to the nearest int

    def write_iv_cache(self):
        """Write the IV cache if we've dirtied it."""
        if self.iv_cache_dirty:
            self.iv_cache.write()

    def optimize_iv(self, max_cp=None, iv_cache=OPTIMAL_IV):
        """Optimize the IV stat product & level for a given CP cap."""
        # static cache for all instances of a pokemon
        if Pokemon.iv_cache is None:
            Pokemon.iv_cache = Cache(iv_cache)
            self.iv_cache_dirty = False
            if Pokemon.iv_cache:
                logging.info("opened cache %s with %d entries", iv_cache, len(Pokemon.iv_cache))

        o = Pokemon.iv_cache.get(self.name, {}).get(str(max_cp))
        if o is not None:
            # update to the optimal cached stats
            self.update(attack=o[0], defense=o[1], stamina=o[2], level=o[3])
            # return the cached value
            return o

        # return max CP if we have no limit, or max IV is below the cap
        self.update(attack=15, defense=15, stamina=15, level=41)
        cp = self.cp()
        if max_cp is None or cp <= max_cp:
            max_cp_s = max_cp is None and "INF" or str(max_cp)
            logging.debug("15/15/15/41=%d, under the %s IV cap", cp, max_cp_s)
            o = [15, 15, 15, 41, self.cp(), self.stat_product(full_precision=True)]
            Pokemon.iv_cache.setdefault(self.name, {})[max_cp] = o
            self.iv_cache_dirty = True
            return o

        # return False if the minimum IV is above the limit
        self.update(attack=0, defense=0, stamina=0, level=1)
        cp = self.cp()
        sp = None
        optimal = []
        if cp > max_cp:
            logging.debug("0/0/0/1=%d exceeds the %d IV cap", cp, max_cp)
            Pokemon.iv_cache.setdefault(self.name, {})[max_cp] = False
            self.iv_cache_dirty = True
            return False
        # find the optimal IV/level combination
        for a in range(0, 16):
            for d in range(0, 16):
                for s in range(0, 16):
                    for l in [x * 0.5 for x in range(2, 82)]:
                        self.update(attack=a, defense=d, stamina=s, level=l)
                        t_cp = self.cp()
                        if t_cp > max_cp:
                            break
                        t_sp = self.stat_product(full_precision=True)
                        if sp is None or t_sp > sp:
                            sp = t_sp
                            optimal = [(a, d, s, l, t_cp, t_sp)]
                        elif t_sp == sp:
                            optimal.append((a, d, s, l, t_cp, t_sp))
        if optimal:
            o = optimal[0]
            # update the stats with the optimal values
            self.update(attack=o[0], defense=o[1], stamina=o[2], level=o[3])
            # cache the result for the next time
            Pokemon.iv_cache.setdefault(self.name, {})[max_cp] = o
            self.iv_cache_dirty = True
            return o
        return False

    def move_combinations(self):
        return self.gm.move_combinations(self.name)

    def optimize_moves(self, target=None):
        """Optimize the move set for a given target."""
        combos = {}
        for fname in self.possible_fast:
            fast = Move(self, fname)
            for cname in self.possible_charged:
                charged = Move(self, cname)
                t = (1.0 + fast.cooldown)
                # damage per turn
                dpt = fast.damage(target) / t
                # energy per turn
                ept = fast.energy_delta / t
                # charged damage per energy
                cdpe = charged.damage(target) / -charged.energy_delta
                # add charged damage per turn
                dpt += cdpe * ept
                result = (fast, charged)
                combos.setdefault(dpt, []).append(result)
        results = sorted(combos.keys(), reverse=True)
        best = combos[results[0]]
        fast = best[0][0]
        charged = [best[0][1]]
        if len(best) > 1:
            # Add the other best move
            charged.append(best[1][1])
        else:
            for result in results[1:]:
                for combo in combos[result]:
                    charged2 = combo[1]
                    if charged2.name != charged[0].name:
                        charged.append(charged2)
                        break
                if len(charged) > 1:
                    break
        # update to the best fast/charged moves
        self.update(fast=fast, charged=charged)


class PokemonUnitTest(unittest.TestCase):
    gm = None
    _time = 0.5

    def setUp(self):
        if not PokemonUnitTest.gm:
            PokemonUnitTest.gm = game_master.GameMaster()

    def test_random_create(self):
        """Create three random monsters."""
        logging.info("3 random monsters:")
        for i in range(3):
            logging.info("%d: %s", i, str(Pokemon(self.gm)))

    def test_eqivalence(self):
        """Test the equivalence operator."""
        kyogreA = Pokemon(self.gm, "KYOGRE", attack=15, defense=15, stamina=15, level=40)
        kyogreB = Pokemon(self.gm, "KYOGRE", attack=15, defense=10, stamina=15, level=40)
        self.assertNotEqual(kyogreA, kyogreB)
        kyogreB.update(defense=15)
        self.assertEqual(kyogreA, kyogreB)
        kyogreB.update(level=39)
        self.assertNotEqual(kyogreA, kyogreB)

    def test_create_speed(self):
        end = time.time() + self._time
        count = 0
        while end > time.time():
            Pokemon(self.gm)
            count += 1
        logging.info("Created %0.0f pokemon/s", count / self._time)

    def test_update_speed(self):
        end = time.time() + self._time
        count = 0
        pokey = Pokemon(self.gm)
        names = list(self.gm.pokemon.keys())
        while end > time.time():
            pokey.update(name=random.choice(names))
            count += 1
        logging.info("Updated %0.0f pokemon/s", count / self._time)

    def test_cp(self):
        """Test that we can compute CP correctly."""
        self.assertEqual(4115, int(Pokemon(self.gm, "KYOGRE", attack=15, defense=15, stamina=15, level=40).cp()))
        self.assertEqual(3741, int(Pokemon(self.gm, "DIALGA", attack=15, defense=15, stamina=14, level=35).cp()))
        self.assertEqual(394, int(Pokemon(self.gm, "SEEL", attack=2, defense=1, stamina=0, level=18).cp()))
        self.assertEqual(92, int(Pokemon(self.gm, "RALTS_SHADOW", attack=0, defense=4, stamina=10, level=8).cp()))
        self.assertEqual(221, int(Pokemon(self.gm, "VENONAT_SHADOW", attack=15, defense=13, stamina=11, level=8).cp()))
        self.assertEqual(256, int(Pokemon(self.gm, "MEOWTH_SHADOW", attack=10, defense=9, stamina=15, level=13).cp()))
        self.assertEqual(1227, int(Pokemon(self.gm, "MOLTRES_SHADOW", attack=5, defense=14, stamina=12, level=13).cp()))

    def test_stat_product(self):
        """Test stat products vs pubished numbers. (low precision)"""
        # https://gostadium.club/pvp/iv?pokemon=Skarmory&max_cp=1500&min_iv=0&att_iv=2&def_iv=12&sta_iv=12
        self.assertEqual(2153046, Pokemon(self.gm, "SKARMORY", attack=0, defense=15, stamina=14, level=27.5).stat_product(full_precision=False))
        # https://gostadium.club/pvp/iv?pokemon=Rattata&max_cp=1500&min_iv=0&att_iv=15&def_iv=15&sta_iv=15
        self.assertEqual(576332, Pokemon(self.gm, "RATTATA", attack=15, defense=15, stamina=15, level=40).stat_product(full_precision=False))

    def test_cache(self):
        TEST_FILE = "tmp.json"
        # make a new cache
        cache = Cache(TEST_FILE)
        # store the values
        cache.update({"foo": 1})
        cache.update({"bar": {"baz": 2}})
        cache["bar"]["taco"] = 3
        # verify that we've stored them correctly
        self.assertEqual(1, cache["foo"])
        self.assertEqual(2, cache["bar"]["baz"])
        self.assertEqual(3, cache["bar"]["taco"])
        # make a new cache
        cache.write()
        del cache
        cache = Cache(TEST_FILE)
        # verify that the values are still there
        self.assertEqual(1, cache["foo"])
        self.assertEqual(2, cache["bar"]["baz"])
        self.assertEqual(3, cache["bar"]["taco"])
        del cache
        os.unlink(TEST_FILE)

    def test_damage(self):
        """Test that we can compute damage correctly."""
        ctx = CONTEXT.COMBAT
        kyogre = Pokemon(self.gm, "KYOGRE", level=40,
                         attack=15, defense=15, stamina=15,
                         fast="WATERFALL_FAST", charged=["SURF"],
                         context=ctx)
        groudon = Pokemon(self.gm, "GROUDON", level=40,
                          attack=15, defense=15, stamina=15,
                          fast="MUD_SHOT_FAST", charged=["EARTHQUAKE"],
                          context=ctx)
        self.assertEqual(18, kyogre.fast.damage(groudon))
        self.assertEqual(3, groudon.fast.damage(kyogre))

    def test_move_combinations(self):
        """Test the move combination iterator."""
        p = Pokemon(self.gm)
        i = 1
        for fast, charged in p.move_combinations():
            p.update(fast=fast, charged=charged)
            logging.info("Move Combination %d: %s", i, p)
            i += 1

    def test_optimize_moves(self):
        """Test that we can compute optimal moves correctly."""
        ctx = CONTEXT.COMBAT
        kyogre = Pokemon(self.gm, "KYOGRE", level=40,
                         attack=15, defense=15, stamina=15,
                         fast="WATERFALL_FAST", charged=["THUNDER"],
                         context=ctx)
        groudon = Pokemon(self.gm, "GROUDON", level=40,
                          attack=15, defense=15, stamina=15,
                          fast="DRAGON_TAIL_FAST", charged=["FIRE_BLAST"],
                          context=ctx)
        kyogre.optimize_moves(groudon)
        logging.info("KvG: %s", kyogre)
        groudon.optimize_moves(kyogre)
        logging.info("GvK: %s", groudon)
        kyogre.optimize_moves(kyogre)
        logging.info("KvK: %s", kyogre)
        groudon.optimize_moves(groudon)
        logging.info("GvG: %s", kyogre)


if __name__ == "__main__":
    """Not intended for standalone use."""
    parser = argparse.ArgumentParser(description='classes for pokemon analysis')
    parser.add_argument("-v", dest="verbose", help="verbose output", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
        log_format = "%(levelname)-5s %(funcName)s:%(lineno)d > %(message)s\n"
    else:
        log_level = logging.INFO
        log_format = "%(message)s"

    logging.basicConfig(level=log_level, format=log_format)

    unittest.main()
