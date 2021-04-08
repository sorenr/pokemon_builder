#!/usr/bin/env python3

import logging
import argparse
import random
import unittest
import enum
import time
import json
import os
import math
import numpy
import re

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
            self.is_legacy = self.name not in self.pokemon.data.get(game_master.GameMaster.K_FAST)
        else:
            self.bonus_atk = self.pokemon.settings.get('chargeAttackBonusMultiplier', 1)
            self.is_legacy = self.name not in self.pokemon.data.get(game_master.GameMaster.K_CHARGED)

        self.power = self.data.get('power', 0)
        if self.type in self.pokemon.type:
            self.stab = self.pokemon.settings['sameTypeAttackBonusMultiplier']
        else:
            self.stab = 1

    def damage(self, target):
        """Calculate damage dealt to the target."""
        if not self.power:
            return 1
        attack = self.pokemon.attack * self.bonus_atk * self.pokemon.bonus_atk
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

        # apply the attack damage
        target.hp -= damage
        self.pokemon.energy += self.energy_delta
        self.pokemon.cooldown -= 1 + self.cooldown
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

    def __repr__(self):
        return "'{0:s}'".format(self.name)

    def __str__(self):
        """Return the attack name, minus the suffix."""
        suffix = self.is_legacy and '*' or ''
        if self.name.endswith(self._FSUFFIX):
            return self.name[:-self._FL] + suffix
        return self.name + suffix


class Cache(dict):
    """A dict which persists."""
    def __init__(self, path):
        self.dirty = False
        self.path = path
        self.dirty = False
        try:
            with open(path) as fd:
                self.update(json.loads(fd.read()))
            logging.debug("Loaded %d entries from %s", len(self), self.path)
            self.fixup(self)
        except FileNotFoundError:
            pass

    def fixup(self, loc):
        """Convert 'null':value into None:value."""
        for k in list(loc.keys()):
            v = loc[k]
            try:
                ke = eval(k)
            except:
                pass
            else:
                loc[ke] = loc[k]
                del loc[k]
            if isinstance(v, dict):
                self.fixup(v)

    def write_prep(self, loc):
        """Prepare a dict for writing to json."""
        rv = {}
        for k, v in loc.items():
            if isinstance(v, dict):
                v = self.write_prep(v)
            rv[str(k)] = v
        return rv

    def write(self, newpath=None):
        """Write the cache before we exit."""
        if not newpath:
            newpath = self.path
        with open(newpath, 'w') as fd:
            json.dump(self.write_prep(self), fd, sort_keys=True)
        logging.info("Wrote %d entries to %s", len(self), self.path)
        self.dirty = False
        return newpath


class Pokemon():
    """Full representation of a specific Pokemon: type, IVs, attacks."""

    fast = None
    charged = []
    iv_cache = None
    _iv_setup = False

    def __init__(self, gm,
                 name=VAL.RANDOM,
                 attack=VAL.RANDOM,
                 defense=VAL.RANDOM,
                 stamina=VAL.RANDOM,
                 level=VAL.RANDOM,
                 fast=VAL.RANDOM,
                 charged=VAL.RANDOM,
                 context=CONTEXT.COMBAT,
                 state=None,
                 iv_cache=OPTIMAL_IV):
        if Pokemon.iv_cache is None:
            Pokemon.iv_cache = Cache(iv_cache)
            if len(Pokemon.iv_cache) > 0:
                logging.info("opened cache %s with %d entries", iv_cache, len(Pokemon.iv_cache))
        self.gm = gm

        # string describing the state of a pokemon
        # CHARMANDER:EMBER_FAST,FLAMETHROWER+FLAME_BURST
        if type(state) == str:
            parts = re.split(r'[:,+]', state)
            name, charged_s = parts[0], parts[1:]
            fast_s = [x for x in parts if x.endswith("_FAST")]
            if fast_s:
                assert 1 == len(fast_s)
                fast = fast_s[0]
                charged_s.remove(fast)
            if charged_s:
                charged = charged_s

        if type(state) == tuple:
            name, attack, defense, stamina, level, fast, charged = state

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
            self.name_unique = self.data[game_master.GameMaster.K_ID_UNIQUE]
            self.stats = self.data[game_master.GameMaster.K_STATS]
            self.type = {self.data[game_master.GameMaster.K_TYPE1], self.data.get(game_master.GameMaster.K_TYPE2)}
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
            level = 0.5 * random.randint(2, game_master.GameMaster.K_LEVEL_MAX * 2)
        elif level is VAL.OPTIMAL:
            level = game_master.GameMaster.K_LEVEL_MAX
        if level is not VAL.DONT_SET:
            self.level = level
            self.cpm = self.gm.cp_multiplier(self.level)

        if attack is VAL.RANDOM:
            attack = random.randint(0, 15)
        elif attack is VAL.OPTIMAL:
            attack = 15
        if attack is not VAL.DONT_SET:
            self.iv_attack = attack

        if defense is VAL.RANDOM:
            defense = random.randint(0, 15)
        elif defense is VAL.OPTIMAL:
            defense = 15
        if defense is not VAL.DONT_SET:
            self.iv_defense = defense

        if stamina is VAL.RANDOM:
            stamina = random.randint(0, 15)
        elif stamina is VAL.OPTIMAL:
            stamina = 15
        if stamina is not VAL.DONT_SET:
            self.iv_stamina = stamina

        self.attack = (self.iv_attack + self.stats[self.gm.K_BASE_ATTACK]) * self.cpm
        self.defense = (self.iv_defense + self.stats[self.gm.K_BASE_DEFENSE]) * self.cpm
        self.stamina = (self.iv_stamina + self.stats[self.gm.K_BASE_STAMINA]) * self.cpm

        self.defense *= self.settings.get(self.gm.K_BONUS_DEF, 1)

        # A move's stats depend on its pokemon, so set the pokemon stats before setting its moves.
        if fast is VAL.RANDOM:
            fast = random.choice(list(self.possible_fast))
        if fast not in [VAL.DONT_SET, VAL.OPTIMAL]:
            self.fast = isinstance(fast, Move) and fast or Move(self, fast)

        if charged is VAL.RANDOM:
            # pick 1 or 2 charged attacks
            p_charged = list(self.possible_charged)
            charged = [random.choice(p_charged)]
            if len(p_charged) > 1 and random.random() > 0.5:
                p_charged.remove(charged[0])
                charged.append(random.choice(p_charged))
        if charged not in [VAL.DONT_SET, VAL.OPTIMAL]:
            # make charged Moves if it's not a Move already
            self.charged = [isinstance(c, Move) and c or Move(self, c) for c in charged]

        # reset to default combat values
        self.reset()

        # pick optimal moves if requested
        if fast is VAL.OPTIMAL or charged is VAL.OPTIMAL:
            self.optimize_moves(target=None, fast=fast is VAL.OPTIMAL, charged=charged is VAL.OPTIMAL)

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

    def copy(self):
        return Pokemon(self.gm,
                       name=self.name,
                       attack=self.iv_attack,
                       defense=self.iv_defense,
                       stamina=self.iv_stamina,
                       level=self.level,
                       fast=self.fast.name,
                       charged=[x.name for x in self.charged])

    def __str__(self):
        """Return a human-readable string representation of the pokemon."""
        iv_str = "{0:d}/{1:d}/{2:d}".format(self.iv_attack, self.iv_defense, self.iv_stamina)
        type_str = "/".join(sorted([str(x) for x in self.type]))
        moves_str = str(self.fast) + " " + "+".join(sorted([str(x) for x in self.charged]))
        cp = self.cp()
        return f"{self.name:s} ({type_str:s}) {iv_str:s} {self.level:0.1f} {moves_str:s} CP={cp:0.1f}"

    def __repr__(self):
        args = [self.name,
                self.fast, self.charged,
                self.iv_attack, self.iv_defense, self.iv_stamina,
                self.level]
        args = [repr(x) for x in args]
        args = ", ".join(args)
        return "{0:s}({1:s})".format(self.__class__.__name__, args)

    def __eq__(self, b):
        """Return whether we're comparing two identical pokemon."""
        return (self.name == b.name and
                self.iv_attack == b.iv_attack and
                self.iv_defense == b.iv_defense and
                self.iv_stamina == b.iv_stamina and
                self.level == b.level)

    def __iter__(self):
        """Convert this Pokemon into a list or tuple."""
        vals = (
            self.name,
            self.iv_attack,
            self.iv_defense,
            self.iv_stamina,
            self.level,
            self.fast.name,
            tuple(sorted([x.name for x in self.charged]))
        )
        for i in vals:
            yield i

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

    def optimize_iv_serial(self, cp_max, full_precision=True):
        """Optimize IVs one combination at a time."""
        sp_optimal = None
        cp_optimal = None
        optimal = []
        steps = range(2, 1 + 2 * self.gm.K_LEVEL_MAX)
        steps = [x * 0.5 for x in steps]
        for atk in range(0, 16):
            for defn in range(0, 16):
                for sta in range(0, 16):
                    for lev in steps:
                        self.update(attack=atk, defense=defn, stamina=sta, level=lev)
                        cp_t = self.cp()
                        # int truncation means cp=2500.99 still counts as 2500
                        if int(cp_t) > int(cp_max):
                            break
                        sp_t = self.stat_product(full_precision=full_precision)
                        if (sp_optimal is None or sp_t > sp_optimal or
                                (sp_t == sp_optimal and cp_t > cp_optimal)):
                            sp_optimal = sp_t
                            cp_optimal = cp_t
                            optimal = [(atk, defn, sta, lev, cp_t, sp_t)]
                        elif sp_t == sp_optimal:
                            optimal.append((atk, defn, sta, lev, cp_t, sp_t))
        return optimal

    def optimize_iv_simd(self, cp_max, full_precision=True):
        """Compute the optimal IV stat product using scipy arrays."""
        # make static arrays if it's the first time
        if not Pokemon._iv_setup:
            ivs = numpy.arange(16, dtype=numpy.float32)
            steps = range(2, 1 + 2 * self.gm.K_LEVEL_MAX)
            cpm = numpy.array([self.gm.cp_multiplier(x * 0.5) for x in steps])
            Pokemon._iv_adsl = numpy.array(numpy.meshgrid(ivs, ivs, ivs, cpm)).T.reshape(-1, 4)
            Pokemon._iv_ads = Pokemon._iv_adsl[:, 0:3]
            Pokemon._cpm = Pokemon._iv_adsl[:, -1]
            Pokemon._cpm = Pokemon._cpm.reshape(Pokemon._cpm.shape[0], 1)
            Pokemon._iv_setup = True

        # compute base attack/defense/stamina
        ads = Pokemon._iv_ads + numpy.array([
            self.stats[self.gm.K_BASE_ATTACK],
            self.stats[self.gm.K_BASE_DEFENSE],
            self.stats[self.gm.K_BASE_STAMINA]])

        # compute cp
        ads *= Pokemon._cpm

        # 0.1 * attack * sqrt(defense) * sqrt(stamina)
        cp = 0.1 * ads[:, 0] * numpy.prod(numpy.power(ads[:, 1:3], 0.5), axis=1)

        # extract cp < cp_max
        i_good = numpy.extract(cp < cp_max + 1, numpy.arange(len(cp)))
        adsl = Pokemon._iv_adsl[i_good]
        ads = ads[i_good]
        cp = cp[i_good]

        # compute stat product
        if full_precision:
            sp = numpy.prod(ads, axis=1)
        else:
            # multiply defense and stamina as floats
            sp = numpy.prod(ads[:, :2], axis=1)
            # multiply attack cast as int
            sp *= ads[:, 2].astype(dtype=numpy.uint16)
            # round to the nearest number and cast as int
            sp = numpy.around(sp).astype(numpy.uint32)

        # filter results with max stat product
        i_max = numpy.nonzero(sp == numpy.max(sp))[0]
        o = adsl[i_max]
        cp = cp[i_max]
        sp = sp[i_max]

        # filter results with same stat product, max cp
        i_max = numpy.nonzero(cp == numpy.max(cp))[0]
        o = o[i_max]
        cp = cp[i_max]
        sp = sp[i_max]

        if full_precision:
            sp = [float(x) for x in sp]
        else:
            sp = [int(x) for x in sp]

        optimal = []
        for i, row in enumerate(o):
            optimal.append([
                int(row[0]),
                int(row[1]),
                int(row[2]),
                self.gm.cpm_level(row[3]),
                float(cp[i]),
                sp[i]
            ])
        return optimal

    def optimize_iv(self, cp_max, simd=True, full_precision=True):
        """Optimize the IV stat product & level for a given CP cap."""
        try:
            o = Pokemon.iv_cache[self.name][cp_max]
            if o is not None:
                # update to the optimal cached stats
                self.update(attack=o[0], defense=o[1], stamina=o[2], level=o[3])
                # return the cached value
                return o
        except KeyError:
            pass

        # return max CP if we have no limit, or max IV is below the cap
        self.update(attack=15, defense=15, stamina=15, level=self.gm.K_LEVEL_MAX)
        cp = self.cp()
        o = [15, 15, 15, self.gm.K_LEVEL_MAX, self.cp(), self.stat_product(full_precision=full_precision)]
        if cp_max is None:
            return o
        if cp <= cp_max:
            logging.debug("%s 15/15/15/%d=%d, under the %s IV cap", self.name, self.level, cp, cp_max)
            Pokemon.iv_cache.setdefault(self.name, {})[cp_max] = o
            Pokemon.iv_cache.dirty = True
            return o

        # return False if the minimum IV is above the limit
        self.update(attack=0, defense=0, stamina=0, level=1)
        cp = self.cp()
        if cp_max is not None and cp > cp_max:
            logging.debug("%s 0/0/0/1=%d exceeds the %d IV cap", self.name, cp, cp_max)
            Pokemon.iv_cache.setdefault(self.name, {})[cp_max] = False
            Pokemon.iv_cache.dirty = True
            return False

        # find the optimal IV/level combination
        if simd:
            optimal = self.optimize_iv_simd(cp_max=cp_max, full_precision=full_precision)
        else:
            optimal = self.optimize_iv_serial(cp_max=cp_max, full_precision=full_precision)

        if optimal:
            # select the first optimal result from the sorted list
            o = sorted(optimal)[0]
            # update the stats with the optimal values
            self.update(attack=o[0], defense=o[1], stamina=o[2], level=o[3])
            # cache the result for the next time
            Pokemon.iv_cache.setdefault(self.name, {})[cp_max] = o
            Pokemon.iv_cache.dirty = True
            return o
        return False

    def move_combinations(self, best=False, r=2):
        return self.gm.move_combinations(self.name, best=best, r=r)

    def dpt(self, fast, charged, target=None):
        """damage per turn against the given target"""
        t = (1.0 + fast.cooldown)
        # damage per turn
        dpt = fast.damage(target) / t
        # energy per turn
        ept = fast.energy_delta / t
        # charged damage per energy
        cdpe = charged.damage(target) / -charged.energy_delta
        # add charged damage per turn
        return dpt + cdpe * ept

    def ttk(self, fast, charged, target, shields=0):
        """turns to kill the given target"""
        self.reset()
        target.reset()
        turns = 0
        while target.hp > 0:
            # charged attack if it's ready
            if self.energy >= -charged.energy_delta:
                charged.attack(target, shield=shields >= 1)
                shields = max(shields - 1, 0)
                turns += 1
            # otherwise fast attack
            else:
                damage_fast = fast.damage(target)
                # turns to next charged attack
                next_charged = (-charged.energy_delta - self.energy) / fast.energy_delta
                # turns to kill with only fast attacks
                next_kill = target.hp / damage_fast
                # perform the next N fast attacks
                fast_attacks = int(math.ceil(min(next_charged, next_kill)))
                target.hp -= fast_attacks * damage_fast
                self.energy += fast_attacks * fast.energy_delta
                turns += fast_attacks * (1 + fast.cooldown)
        return turns

    def optimize_moves(self, target=None, fast=True, charged=True):
        """Optimize the move set for damage per turn against a given target."""
        combos = {}

        # generate possible fast moves
        if fast:
            possible_fast = [Move(self, name) for name in self.possible_fast]
        else:
            possible_fast = [self.fast]

        # generate possible charged moves
        if charged:
            possible_charged = [Move(self, name) for name in self.possible_charged]
        else:
            possible_charged = self.charged

        for move_f in possible_fast:
            for move_c in possible_charged:
                dpt = self.dpt(move_f, move_c, target)
                combo = (move_f, move_c)
                combos.setdefault(dpt, []).append(combo)
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

    def issuperset(self, other):
        """return whether 'self' is functionally the same, or has extra moves."""

        # if it's not the same stats and type it can't be a superset
        if self.stats != other.stats or self.type != other.type:
            return 0

        bonuses = (game_master.GameMaster.K_BONUS_DEF,
                   game_master.GameMaster.K_SHADOW_BONUS_DEF,
                   game_master.GameMaster.K_SHADOW_BONUS_ATK)
        for bonus in bonuses:
            if self.data.get(bonus, 1.0) != other.data.get(bonus, 1.0):
                return 0

        if self.possible_fast == other.possible_fast and self.possible_charged == other.possible_charged:
            # identical supersets
            return 1

        if not self.possible_fast.issuperset(other.possible_fast):
            return 0
        if not self.possible_charged.issuperset(other.possible_charged):
            return 0

        # is a superset containing extra items
        return 2


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

    def test_ttk_lucario_tyranitar(self):
        lucario = Pokemon(self.gm, "LUCARIO", level=40,
                          attack=15, defense=15, stamina=15,
                          fast="COUNTER_FAST", charged=["POWER_UP_PUNCH"],
                          context=CONTEXT.COMBAT)
        tyranitar = Pokemon(self.gm, "TYRANITAR", level=40,
                            attack=15, defense=15, stamina=15,
                            fast="SMACK_DOWN_FAST", charged=["CRUNCH"],
                            context=CONTEXT.COMBAT)
        self.assertEqual(21, lucario.ttk(fast=lucario.fast, charged=lucario.charged[0], target=tyranitar, shields=1))
        self.assertEqual(41, tyranitar.ttk(fast=tyranitar.fast, charged=tyranitar.charged[0], target=lucario, shields=1))

    def test_ttk_lucario_snorlax(self):
        lucario = Pokemon(self.gm, "LUCARIO", level=40,
                          attack=15, defense=15, stamina=15,
                          fast="COUNTER_FAST", charged=["POWER_UP_PUNCH"],
                          context=CONTEXT.COMBAT)
        snorlax = Pokemon(self.gm, "SNORLAX", level=40,
                          attack=15, defense=15, stamina=15,
                          fast="LICK_FAST", charged=["BODY_SLAM"],
                          context=CONTEXT.COMBAT)
        self.assertEqual(30, lucario.ttk(fast=lucario.fast, charged=lucario.charged[0], target=snorlax, shields=1))
        self.assertEqual(38, snorlax.ttk(fast=snorlax.fast, charged=snorlax.charged[0], target=lucario, shields=1))

    def assertEqualParts(self, tuple1, tuple2, delta=None):
        """Like assertTupleEqual except special rules for float members."""
        self.assertEqual(len(tuple1), len(tuple2))
        for val1, val2 in zip(tuple1, tuple2):
            self.assertEqual(type(val1), type(val2))
            if isinstance(val1, float):
                self.assertAlmostEqual(val1, val2, delta=delta)
            else:
                self.assertEqual(val1, val2)

    def test_iv_high(self):
        """Check that levels >40 calculate correct values."""
        self.assertEqual(2926, int(Pokemon(self.gm, name="CHARIZARD", attack=15, defense=15, stamina=15, level=41).cp()))
        self.assertEqual(3410, int(Pokemon(self.gm, name="EXCADRILL", attack=15, defense=15, stamina=15, level=44).cp()))

    def test_iv_simd(self):
        """Check that iv_serial and iv_simd calculate the same values."""
        logging.info("iv serial/simd tests")
        t_simd = 0
        t_serial = 0
        for i in range(3):
            p = Pokemon(self.gm, attack=0, defense=0, stamina=0, level=1)

            t_serial -= time.time()
            oo = p.optimize_iv_serial(cp_max=2500, full_precision=True)[0]
            t_serial += time.time()
            t_simd -= time.time()
            os = p.optimize_iv_simd(cp_max=2500, full_precision=True)[0]
            t_simd += time.time()
            logging.info("iv_serial %s %s True", p.name, oo)
            logging.info("  iv_simd %s %s True", p.name, os)
            self.assertEqualParts(oo, os, delta=3)

            t_serial -= time.time()
            oo = p.optimize_iv_serial(cp_max=2500, full_precision=False)[0]
            t_serial += time.time()
            t_simd -= time.time()
            os = p.optimize_iv_simd(cp_max=2500, full_precision=False)[0]
            t_simd += time.time()
            logging.info("iv_serial %s %s True", p.name, oo)
            logging.info("  iv_simd %s %s True", p.name, os)
            self.assertEqualParts(oo, os, delta=3)
        logging.info("SIMD %0.1fx faster than serial", t_serial / t_simd)

    def test_state(self):
        # make a pokemon
        p_old = Pokemon(self.gm)

        # make a state description
        state = "{0:s}:{1:s},{2:s}".format(
            p_old.name,
            p_old.fast.name,
            "+".join(x.name for x in p_old.charged)
        )

        # create new pokemon with that state description
        p_new = Pokemon(
            gm=self.gm,
            state=state,  # state defines name/moves
            attack=p_old.iv_attack,
            defense=p_old.iv_defense,
            stamina=p_old.iv_stamina,
            level=p_old.level
        )

        # they should be the same
        self.assertEqual(p_old, p_new)


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

    unittest.main(failfast=True)
