#!/usr/bin/env python3

import unittest
import argparse
import logging
import multiprocessing

import game_master
import pokemon


def attack(attacker, target):
    """Pick an attack (if ready)"""
    # do a fast attack
    if attacker.cooldown >= attacker.fast.cooldown:
        attacker.fast.attack(target)
    else:
        attacker.cooldown += 1

    # for every attack...
    for attack in attacker.charged:
        # if we have enough energy to trigger it...
        if attacker.energy >= -attack.energy_delta:
            damage = attack.damage(target)
            # always shield if the damage would kill the target
            shield = target.hp <= damage
            attack.attack(target, shield=shield, damage=damage)
            target.cooldown -= 1
            return


def combat(p1, p2):
    """Main combat loop."""
    p1.reset()
    p2.reset()

    # pokemon with the higher attack goes first
    if p2.attack > p1.attack:
        p1, p2 = p2, p1

    logging.debug("p1: %s (%0.1f)", p1.name, p1.attack)
    logging.debug("p2: %s (%0.1f)", p2.name, p2.attack)

    turn = 0
    while True:
        logging.debug("turn %d:", turn)
        attack(p1, p2)
        if p2.hp <= 0:
            logging.debug("%s wins with %0.1f hp", p1.name, p1.hp)
            return p1
        attack(p2, p1)
        if p1.hp <= 0:
            logging.debug("%s wins with %0.1f hp", p2.name, p2.hp)
            return p2
        turn += 1


def rate(args):
    gm, names, opponent, max_cp = args
    p = pokemon.Pokemon(gm)

    results = {}

    for name in names:
        p.update(name)
        p.optimize_iv(max_cp=max_cp)
        for fast, charged in p.move_combinations():
            p.update(fast=fast, charged=charged)
            winner = combat(p, opponent)
            result = winner.hp
            if winner is opponent:
                result = -result
            results.setdefault(result, []).append(str(p))

    return results


def chunk(lst, n):
    """Divide a list into n-sized chunks."""
    n = max(1, n)
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def counter(gm, target, max_cp=None, threads=None):
    if ':' in target:
        target, moves = target.split(":", 1)
        fast, charged = moves.split(",", 1)
        charged = charged.split("+")
        target = pokemon.Pokemon(gm, name=target, fast=fast, charged=charged)
    else:
        target = pokemon.Pokemon(gm, target)

    target.optimize_iv(max_cp=max_cp)
    print("target:", target)

    results = {}
    names = list(gm.pokemon.keys())
    # smeargle has too many move combinations
    names.remove('SMEARGLE')

    # divide the name list into job-sized chunks
    chunk_size = int(len(names)/multiprocessing.cpu_count())
    chunks = chunk(names, chunk_size)

    pool = multiprocessing.Pool(threads)
    for result in pool.imap_unordered(rate, [(gm, names, target, max_cp) for names in chunks]):
        results.update(result)

    for result in sorted(results.keys()):
        for c in results[result]:
            print("{0:0.2f} {1:s}".format(result, c))


class CounterUnitTest(unittest.TestCase):
    gm = None

    def setUp(self):
        if not CounterUnitTest.gm:
            CounterUnitTest.gm = game_master.GameMaster()

    def lucario_v_snorlax(self):
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
            fast="YAWN_FAST",
            charged=["BODY_SLAM"])
        self.assertEqual(2499, int(snorlax.cp()))

        winner = combat(lucario, snorlax)
        # lucario wins...
        self.assertIs(winner, lucario)
        # ...with 105 hp left.
        self.assertEqual(105, int(winner.hp))

    def mime_v_tyranitar(self):
        mime = pokemon.Pokemon(
            self.gm, "MR_MIME", level=23.5,
            attack=14, defense=14, stamina=13,
            fast="ZEN_HEADBUTT_FAST",
            charged=["PSYBEAM"])
        self.assertEqual(1474, int(mime.cp()))

        tyranitar = pokemon.Pokemon(
            self.gm, "TYRANITAR", level=14.5,
            attack=3, defense=11, stamina=11,
            fast="SMACK_DOWN_FAST",
            charged=["CRUNCH"])
        self.assertEqual(1490, int(tyranitar.cp()))

        winner = combat(mime, tyranitar)
        # tyranitar wins...
        self.assertIs(winner, tyranitar)
        # ...with 89 hp left.
        self.assertEqual(89, int(winner.hp))

    def empoleon_v_torterra(self):
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
            charged=["FRENZY_PLANT", "SAND_TOMB"])
        self.assertEqual(1491, int(torterra.cp()))


if __name__ == "__main__":
    """Not intended for standalone use."""
    parser = argparse.ArgumentParser(description='classes for pokemon analysis')
    parser.add_argument("-v", dest="verbose", help="verbose output", action="store_true")
    parser.add_argument("-t", help="threads", type=int)
    parser.add_argument("--cp", dest="max_cp", type=int, help="cp limit")
    parser.add_argument("opponents", nargs="*", help="opponents to counter")
    args = parser.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
        log_format = "%(levelname)-5s %(funcName)s:%(lineno)d > %(message)s\n"
    else:
        log_level = logging.INFO
        log_format = "%(message)s"

    logging.basicConfig(level=log_level, format=log_format)

    if args.opponents:
        gm = game_master.GameMaster()
        counter(gm, args.opponents[0], max_cp=args.max_cp)
    else:
        unittest.main()
