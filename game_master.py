#!/usr/bin/env python3

import os
import re
import time
import json
import logging
import urllib.request
import subprocess
import unittest


"""
    Retrieve, parse, and normalize data from GAME_MASTER.json
"""

GAME_MASTER_URL = "https://raw.githubusercontent.com/pokemongo-dev-contrib/pokemongo-game-master/master/versions/latest/GAME_MASTER.json"
GAME_MASTER_PATH = GAME_MASTER_URL.rsplit('/', 1)[1]


def get_http(url, dest):
    """Retrieve a file via http"""
    if not os.path.exists(dest):
        return urllib.request.urlretrieve(url, dest)


def get_curl(url, dest):
    """Retrieve a file via curl"""
    subprocess.run(["curl", GAME_MASTER_URL, "-o", GAME_MASTER_PATH], check=True)


class GameMaster():
    """Class for normalizing and retrieving data from GAME_MASTER.json"""

    _re_type = re.compile(r'POKEMON_TYPE_(.+)')
    _re_pokemon = re.compile(r'V(\d+)_POKEMON_(.+)')
    _re_forms = re.compile(r'FORMS_V(\d+)_POKEMON_(.+)')
    _re_move = re.compile(r'V(\d+)_MOVE_(.+)')
    _re_combat_move = re.compile(r'COMBAT_V(\d+)_MOVE_(.+)')

    K_FAST = "quickMoves"
    K_CHARGED = "cinematicMoves"
    K_SMEARGLE = "SMEARGLE"
    K_SMEARGLE_MOVES = "SMEARGLE_MOVES_SETTINGS"

    # item prefixes to ignore
    _TID_IGNORE = [
        'adventure_',
        'AVATAR_',
        'BACKGROUND_',
        'BADGE_',
        'BATTLE_',
        'BELUGA_',  # ???
        'BUDDY_',
        'bundle.',
        'camera_',
        'CHARACTER_',
        'ENCOUNTER_',
        'EX_RAID_',
        'FRIENDSHIP_LEVEL_',
        'general1.',
        'GYM_',
        'IAP_',
        'INVASION_',
        'incenseordinary.',
        'ITEM_',
        'itemleadermap',
        'LPSKU_',
        'LUCKY_POKEMON_SETTINGS',
        'ONBOARDING_V2_SETTINGS',
        'PARTY_RECOMMENDATION_SETTINGS',
        'PLATYPUS_ROLLOUT_SETTINGS',
        'POKECOIN_PURCHASE_DISPLAY_GMT',
        'POKEMON_SCALE_SETTINGS_',
        'POKEMON_UPGRADE_SETTINGS',  # FIXME: add upgrade recommendations
        'QUEST_',
        'sequence_',
        'SPAWN_V',
        'TRAINER_',
        'VS_SEEKER_',
        'WEATHER_AFFINITY_',
        'WEATHER_BONUS_',
    ]

    def __init__(self, url=GAME_MASTER_URL):
        """Parse the game_master file."""
        start = time.time()
        # fetch GAME_MASTER if it doesn't exist
        if not os.path.exists(GAME_MASTER_PATH):
            try:
                get_http(GAME_MASTER_URL, GAME_MASTER_PATH)
            except urllib.request.URLError:
                # sometimes Python's SSL certificates are missing
                get_curl(GAME_MASTER_URL, GAME_MASTER_PATH)
            logging.info("Fetched %s from %s in %0.2fs", GAME_MASTER_PATH, GAME_MASTER_URL, time.time() - start)
        assert os.path.exists(GAME_MASTER_PATH)

        # load the data
        start = time.time()
        with open(GAME_MASTER_PATH, 'r') as fd:
            self._game_master = json.loads(fd.read())

        # normalize the data
        self.normalize_data()
        self.startup = time.time() - start
        logging.info("GAME_MASTER [%s] loaded in %0.01fms", self.timestamp, 1000 * self.startup)

    def _should_ignore(self, tid):
        """Should we ignore this item?"""
        for ignore in self._TID_IGNORE:
            if tid.startswith(ignore):
                return True
        return False

    def normalize_data(self):
        self.effectiveness = {}  # type effectiveness
        self.pokemon = {}        # pokemon[monster_name] = monster_data
        self.forms = {}          # forms[form_name] = alt_forms
        self.moves = {}          # moves[move_name] = move_data
        self.moves_combat = {}   # moves_combat[move_name] = move_data
        self._cp_multiplier = None

        """Reconfigure the data to be more useful."""
        self.timestamp = time.ctime(float(self._game_master['timestampMs']) / 1000)

        type_list = []

        for item in self._game_master['itemTemplate']:
            tid = item['templateId']

            # ignore suffixes
            # if self._should_ignore(tid):
            #     continue

            # get cp multiplier
            if tid == "PLAYER_LEVEL_SETTINGS":
                self._cp_multiplier = item['playerLevel']['cpMultiplier']
                continue

            # save smeargle's moves for processing at the end of this loop
            if tid == "SMEARGLE_MOVES_SETTINGS":
                smeargle_moves = item['smeargleMovesSettings']
                continue

            # add forms to self.forms
            r = self._re_forms.match(tid)
            if r:
                settings = item['formSettings']
                baseName = settings['pokemon']
                for formSettings in settings.get('forms', []):
                    formName = formSettings['form']
                    self.forms.setdefault(baseName, []).append(formName)
                continue

            # make type effectiveness
            r = self._re_type.match(tid)
            if r:
                effectiveness = item['typeEffective']
                attack_type = effectiveness['attackType']
                attack_scalar = effectiveness['attackScalar']
                self.effectiveness[attack_type] = attack_scalar
                type_list.append(attack_type)
                continue

            # store pokemon settings
            r = self._re_pokemon.match(tid)
            if r:
                settings = item.get('pokemon') or item.get('pokemonSettings')
                assert settings
                # form name takes precedence over family name
                name = settings.get('form') or settings.get('uniqueId') or settings.get('pokemonId')
                assert name
                self.pokemon[name] = settings
                continue

            # store move settings
            r = self._re_move.match(tid)
            if r:
                settings = item.get('move') or item.get('moveSettings')
                name = settings['movementId']
                assert name not in self.moves
                self.moves[name] = settings
                continue

            # store combat (pvp) move settings
            r = self._re_combat_move.match(tid)
            if r:
                settings = item['combatMove']
                name = r.group(2)
                assert name not in self.moves_combat
                self.moves_combat[name] = settings
                continue

            # warn about a possibly important item we're ignoring
            # logging.debug("Won't process %s", tid)

        # Add Smeargle's moves
        assert GameMaster.K_FAST not in self.pokemon[GameMaster.K_SMEARGLE]
        assert GameMaster.K_CHARGED not in self.pokemon[GameMaster.K_SMEARGLE]
        self.pokemon[GameMaster.K_SMEARGLE][GameMaster.K_FAST] = smeargle_moves[GameMaster.K_FAST]
        self.pokemon[GameMaster.K_SMEARGLE][GameMaster.K_CHARGED] = smeargle_moves[GameMaster.K_CHARGED]

    def effect(self, attack_type, target_types):
        effect = 1.0
        for ttype in target_types:
            effect *= self.effectiveness[attack_type.value][ttype.value]
        return effect

    def possible_fast(self, name):
        """Return a tuple of fast moves for this pokemon."""
        pdata = self.pokemon[name]
        return pdata.get(GameMaster.K_FAST, [])

    def possible_charged(self, name):
        """Return a tuple of charged) moves for this pokemon."""
        pdata = self.pokemon[name]
        return pdata.get(GameMaster.K_CHARGED, [])

    def report(self):
        """Write a summary of the data we've received."""
        logging.info("EFFECTIVENESS %d", len(self.effectiveness))
        logging.info("POKEMON %d", len(self.pokemon))
        logging.info("FORMS %d", len(self.forms))
        logging.info("MOVES %d", len(self.moves))


class GameMasterUnitTest(unittest.TestCase):
    gm = None

    def setUp(self):
        if not GameMasterUnitTest.gm:
            GameMasterUnitTest.gm = GameMaster()

    def test_report(self):
        self.gm.report()

    def test_attack_num(self):
        """Make sure all pokemon have at least one fast and one charged attack."""
        for name in self.gm.pokemon.keys():
            pfast = self.gm.possible_fast(name)
            pcharged = self.gm.possible_charged(name)
            self.assertGreaterEqual(len(pfast), 1)
            self.assertGreaterEqual(len(pcharged), 1)


if __name__ == "__main__":
    import argparse

    """Not intended for standalone use."""
    parser = argparse.ArgumentParser(description='Retrieve and parse a GAME_MASTER.json file')
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
