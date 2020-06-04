#!/usr/bin/env python3

import game_master
import pokemon

import logging
import argparse
import unittest


def attack_fast(attacker, target):
    """Do a fast attack (if ready)."""
    # do a fast attack
    if attacker.cooldown >= attacker.fast.cooldown:
        attacker.fast.attack(target)


def attack_charged(attacker, target, move, shield):
    """Do a charged attack (if enough energy)."""
    if attacker.energy >= -move.energy_delta:
        damage = move.damage(target)
        move.attack(target, shield=shield, damage=damage)
        return True


def should_shield(shield):
    if isinstance(shield, list) and len(shield):
        return shield.pop()
    return bool(shield)


def combat(p1, p1_attack, p2, p2_attack, turn=0):
    """Main combat loop."""
    while True:
        # pokemon with the higher attack goes first
        if p2.attack > p1.attack:
            p1, p1_attack, p2, p2_attack = p2, p2_attack, p1, p1_attack

        for p in [p1, p2]:
            logging.debug("turn %d: %s hp=%d e=%d cd=%d", turn, p.name, p.hp, p.energy, p.cooldown)

        # each player picks their attacks
        p1_charged = p1.energy >= -p1_attack.energy_delta
        p2_charged = p2.energy >= -p2_attack.energy_delta

        if not p1_charged:
            attack_fast(p1, p2)
        if not p2_charged:
            attack_fast(p2, p1)

        if p1_charged:
            p1 = attack_charged(p1, p2, p1_attack, should_shield(p2_shield))
        if p2_charged:
            p2 = attack_charged(p2, p1, p2_attack, should_shield(p1_shield))

        if p2.hp <= 0:
            logging.debug("%s wins with %0.1f hp", p1.name, p1.hp)
            return p1

        if p1.hp <= 0:
            logging.debug("%s wins with %0.1f hp", p2.name, p2.hp)
            return p2

        turn += 1
        p1.cooldown += 1
        p2.cooldown += 1


class CounterUnitTest(unittest.TestCase):
    gm = None

    def setUp(self):
        if not CounterUnitTest.gm:
            CounterUnitTest.gm = game_master.GameMaster()

    def test_lucario_v_snorlax(self):
        # https://pvpoke.com/battle/2500/lucario-38.5-1-15-15-4-4-1/snorlax-29.5-0-12-15-4-4-1/11/1-4-0/1-1-0/
        lucario = pokemon.Pokemon(
            self.gm, "LUCARIO", level=38.5,
            attack=1, defense=15, stamina=15,
            fast="COUNTER_FAST",
            charged=["POWER_UP_PUNCH"])
        self.assertEqual(2497, int(lucario.cp()))

        snorlax = pokemon.Pokemon(
            self.gm, "SNORLAX", level=29.5,
            attack=0, defense=12, stamina=15,
            fast="LICK_FAST",
            charged=["BODY_SLAM"])
        self.assertEqual(2499, int(snorlax.cp()))
        snorlax.reset(shields=[True, False])

        # simulate combat
        winner = combat(
            lucario, lucario.charged[0],
            snorlax, snorlax.charged[0])

        # lucario wins...
        self.assertIs(winner, lucario)
        # ...with 105 hp left.
        self.assertEqual(34, int(winner.hp))

    def test_mime_v_tyranitar(self):
        mime = pokemon.Pokemon(
            self.gm, "MR_MIME", level=23.5,
            attack=14, defense=14, stamina=13,
            fast="ZEN_HEADBUTT_FAST",
            charged=["PSYBEAM"])
        self.assertEqual(1474, int(mime.cp()))

        # simulate combat
        tyranitar = pokemon.Pokemon(
            self.gm, "TYRANITAR", level=14.5,
            attack=3, defense=11, stamina=11,
            fast="SMACK_DOWN_FAST",
            charged=["CRUNCH"])
        self.assertEqual(1490, int(tyranitar.cp()))

        # simulate combat
        winner = combat(
            mime, mime.charged[0],
            tyranitar, tyranitar.charged[0])

        # tyranitar wins...
        self.assertIs(winner, tyranitar)
        # ...with 89 hp left.
        self.assertEqual(89, int(winner.hp))

    def test_empoleon_v_torterra(self):
        empoleon = pokemon.Pokemon(
            self.gm, "EMPOLEON", level=17,
            attack=15, defense=14, stamina=9,
            fast="METAL_CLAW_FAST",
            charged=["HYDRO_CANNON"])
        self.assertEqual(1385, int(empoleon.cp()))

        torterra = pokemon.Pokemon(
            self.gm, "TORTERRA", level=19.5,
            attack=0, defense=12, stamina=9,
            fast="RAZOR_LEAF_FAST",
            charged=["FRENZY_PLANT"])
        self.assertEqual(1491, int(torterra.cp()))

        # simulate combat
        winner = combat(
            empoleon, empoleon.charged[0],
            torterra, torterra.charged[0])

        # tyranitar wins...
        self.assertIs(winner, torterra)
        # ...with 89 hp left.
        self.assertEqual(76, int(winner.hp))


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