#!/usr/bin/env python3

import logging
import argparse
import random
import unittest
import enum
import time

import game_master


class VAL(enum.Enum):
    """Actions for setting pokemon attributes."""
    DONT_SET = enum.auto()  # don't do anything
    RANDOM = enum.auto()    # set a random value
    OPTIMAL = enum.auto()   # set an optimal value


class CONTEXT(enum.Enum):
    """Contexts for evaluating an interaction"""
    BATTLE = enum.auto()  # raid
    COMBAT = enum.auto()  # pvp


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
        self.data = self.pokemon.moves[self.name]
        self.type = game_master.Types[self.data['type']]

    def __str__(self):
        """Return the attack name, minus the suffix."""
        if self.name.endswith(self._FSUFFIX):
            return self.name[:-self._FL]
        return self.name


class Pokemon():
    """Full representation of a specific Pokemon: type, IVs, attacks."""

    fast = None
    charged = []

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
            self.data = self.gm.pokemon[self.name]
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

        # A move's stats depend on its pokemon, so set the pokemon stats before setting its moves.
        if fast is VAL.RANDOM:
            fast = random.choice(self.possible_fast)
        if fast is not VAL.DONT_SET:
            self.fast = Move(self, fast)

        if charged is VAL.RANDOM:
            # pick 1 or 2 charged attacks
            p_charged = self.possible_charged.copy()
            charged = [random.choice(p_charged)]
            if len(p_charged) > 1 and random.random() > 0.5:
                p_charged.remove(charged[0])
                charged.append(random.choice(p_charged))
        if charged is not VAL.DONT_SET:
            self.charged = [Move(self, x) for x in charged]

    def cp(self):
        # CP = (Attack * Defense^0.5 * Stamina^0.5 * CP_Multiplier^2) / 10
        # https://gamepress.gg/pokemongo/pokemon-stats-advanced#cp-multiplier
        rv = self.attack
        rv *= pow(self.defense, 0.5)
        rv *= pow(self.stamina, 0.5)
        return rv / 10

    def __str__(self):
        """Return a human-readable string representation of the pokemon."""
        iv_str = "{0:d}/{1:d}/{2:d}".format(self.iv_attack, self.iv_defense, self.iv_stamina)
        type_str = "/".join(sorted([str(x) for x in self.type]))
        moves_str = str(self.fast) + " " + "+".join([str(x) for x in self.charged])
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

    def test_stat_product(self):
        """Test stat products vs pubished numbers. (low precision)"""
        # https://gostadium.club/pvp/iv?pokemon=Skarmory&max_cp=1500&min_iv=0&att_iv=2&def_iv=12&sta_iv=12
        self.assertEqual(2153046, Pokemon(self.gm, "SKARMORY", attack=0, defense=15, stamina=14, level=27.5).stat_product(full_precision=False))
        # https://gostadium.club/pvp/iv?pokemon=Rattata&max_cp=1500&min_iv=0&att_iv=15&def_iv=15&sta_iv=15
        self.assertEqual(576332, Pokemon(self.gm, "RATTATA", attack=15, defense=15, stamina=15, level=40).stat_product(full_precision=False))


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
