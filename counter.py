#!/usr/bin/env python3

import unittest
import argparse
import logging
import multiprocessing
import re

import game_master
import pokemon


def attack_fast(attacker, target):
    """Do a fast attack (if ready)."""
    # do a fast attack
    if attacker.cooldown >= attacker.fast.cooldown:
        attacker.fast.attack(target)
    else:
        attacker.cooldown += 1


def attack_charged(attacker, target, move):
    """Do a charged attack (if enough energy)."""
    if attacker.energy >= -move.energy_delta:
        damage = move.damage(target)
        # always shield if the damage would kill the target
        shield = target.hp <= damage
        move.attack(target, shield=shield, damage=damage)
        return True


def best_move(attacker, target):
    """Pick the best attack for the target."""
    if len(attacker.charged) == 1:
        return attacker.charged[0]
    move_best = None
    dpe_best = None
    for move in attacker.charged:
        damage = move.damage(target)
        dpe = damage/-move.energy_delta
        if dpe_best is None or dpe_best < dpe:
            move_best = move
            dpe_best = dpe
    return move_best


def combat(p1, attack_p1, p2, attack_p2):
    """Main combat loop."""
    p1.reset()
    p2.reset()

    # pokemon with the higher attack goes first
    if p2.attack > p1.attack:
        p1, attack_p1, p2, attack_p2 = p2, attack_p2, p1, attack_p1

    logging.debug("p1: %s (%0.1f)", p1.name, p1.attack)
    logging.debug("p2: %s (%0.1f)", p2.name, p2.attack)

    turn = 0
    while True:
        turn += 1
        logging.debug("turn %d:", turn)

        attack_fast(p1, p2)
        if p2.hp <= 0:
            logging.debug("%s wins with %0.1f hp", p1.name, p1.hp)
            return p1

        attack_fast(p2, p1)
        if p1.hp <= 0:
            logging.debug("%s wins with %0.1f hp", p2.name, p2.hp)
            return p2

        if attack_charged(p1, p2, attack_p1):
            if p2.hp <= 0:
                logging.debug("%s wins with %0.1f hp", p1.name, p1.hp)
                return p1
            # p1 did a charged attack, and p2 got another fast hit so this turn ends
            continue

        attack_charged(p2, p1, attack_p2)
        if p1.hp <= 0:
            logging.debug("%s wins with %0.1f hp", p2.name, p2.hp)
            return p2


class Ranker():
    """Copute fitness for a set of targets."""
    def __init__(self, gm, opponent_names, max_cp=None):
        self.gm = gm
        self.max_cp = max_cp
        self.p = pokemon.Pokemon(gm)
        self.opponents = []

        # generate the target list to battle against all possible challengers
        for opponent_name in [x.upper() for x in opponent_names]:
            parts = re.split(r'[:,+]', opponent_name)
            opponent_name, charged = parts[0], parts[1:]
            fast = [x for x in parts if x.endswith("_FAST")]
            if fast:
                assert 1 == len(fast)
                fast = fast[0]
                charged.remove(fast)
            # select the best general attacks if no specific attack is selected
            else:
                fast = pokemon.VAL.OPTIMAL
            if not charged:
                charged = pokemon.VAL.OPTIMAL
            opponent = pokemon.Pokemon(gm, name=opponent_name, fast=fast, charged=charged)
            assert opponent.optimize_iv(max_cp=max_cp)
            self.opponents.append(opponent)

    def rank_combat(self, names):
        """Rank pokemon listed in 'names' against the given targets."""
        results = {}

        for name in names:
            self.p.update(name)
            self.p.optimize_iv(max_cp=self.max_cp)
            for fast, charged in self.p.move_combinations():
                self.p.update(fast=fast, charged=charged)
                results_t = []
                for opponent in self.opponents:
                    p_best = best_move(p, opponent)
                    t_best = best_move(opponent, p)
                    winner = combat(p, p_best, opponent, t_best)
                    result = winner.hp
                    if winner is opponent:
                        result = -result
                    results_t.append(result)
                results.setdefault(tuple(results_t), []).append(str(p))

        return results

    def rank_ttk(self, names):
        """Rank pokemon listed in 'names' against the turns required to kill each other."""
        results = {}

        for name in names:
            self.p.update(name)
            self.p.optimize_iv(max_cp=self.max_cp)
            for fast, charged in self.p.move_combinations():
                self.p.update(fast=fast, charged=charged)
                results_t = []
                for opponent in self.opponents:
                    p_best = best_move(self.p, opponent)
                    o_best = best_move(opponent, self.p)
                    ttk_o = self.p.ttk(self.p.fast, p_best, opponent)
                    ttk_p = opponent.ttk(opponent.fast, o_best, self.p)
                    result = ttk_p - ttk_o
                    results_t.append(result)
                results.setdefault(tuple(results_t), []).append(str(self.p))

        return results


def chunk(lst, n):
    """Divide a list into n-sized chunks."""
    n = max(1, n)
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def counter(gm, opponent_names, max_cp=None, threads=1):
    ranker = Ranker(gm, opponent_names, max_cp=max_cp)

    team_names = list(gm.pokemon.keys())
    # smeargle has too many move combinations
    team_names.remove('SMEARGLE')

    rank_func = ranker.rank_ttk
    # rank_func = ranker.rank_combat

    if threads == 1:
        # rank the opponents in a single thread
        results = rank_func(team_names)
    else:
        results = {}
        # divide the name list into job-sized chunks
        chunk_size = int(len(team_names) / threads)
        team_name_chunks = chunk(team_names, chunk_size)
        pool = multiprocessing.Pool(threads)
        for result_t in pool.imap_unordered(rank_func, team_name_chunks):
            for bouts, monstas in result_t.items():
                # sum sqrt because for multiple targets winning one
                # by a lot matters less than winning all by a little
                rating = sum([x > 0 and pow(x, 0.5) for x in bouts])
                results.setdefault(rating, {}).setdefault(bouts, []).extend(monstas)

    for rank in sorted(results.keys()):
        bouts_h = results[rank]
        for bout in sorted(bouts_h.keys()):
            bout_str = " ".join(["{0:0.1f}".format(x) for x in bout])
            monstas = bouts_h[bout]
            for monsta in monstas:
                print("{:0.2f} [{:s}] {:s}".format(rank, bout_str, monsta))

    print()
    # print the targets we've ranked against
    for opponent in ranker.opponents:
        print(str(opponent))


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

        winner = combat(lucario, lucario.charged[0], snorlax, snorlax.charged[0])
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
    parser.add_argument("-t", dest="threads", type=int, default=multiprocessing.cpu_count())
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
        counter(gm, args.opponents, max_cp=args.max_cp, threads=args.threads)
    else:
        unittest.main()
