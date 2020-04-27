#!/usr/bin/env python3

import os
import re
import time
import json
import logging
import urllib.request
import subprocess
import unittest
import enum
import itertools
import difflib
import fnmatch


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


class Types(enum.Enum):
    """Index order of POKEMON_TYPE_* effectiveness in GAME_MASTER"""
    POKEMON_TYPE_NORMAL = 0
    POKEMON_TYPE_FIGHTING = 1
    POKEMON_TYPE_FLYING = 2
    POKEMON_TYPE_POISON = 3
    POKEMON_TYPE_GROUND = 4
    POKEMON_TYPE_ROCK = 5
    POKEMON_TYPE_BUG = 6
    POKEMON_TYPE_GHOST = 7
    POKEMON_TYPE_STEEL = 8
    POKEMON_TYPE_FIRE = 9
    POKEMON_TYPE_WATER = 10
    POKEMON_TYPE_GRASS = 11
    POKEMON_TYPE_ELECTRIC = 12
    POKEMON_TYPE_PSYCHIC = 13
    POKEMON_TYPE_ICE = 14
    POKEMON_TYPE_DRAGON = 15
    POKEMON_TYPE_DARK = 16
    POKEMON_TYPE_FAIRY = 17

    def __lt__(self, other):
        if isinstance(other, Types):
            return self.name < other.name
        return self.value < int(other.value)

    def __str__(self):
        return self.name[13:]


TYPE_LIST = sorted(list(Types), key=lambda t: t.value)


class PokemonKeyError(KeyError):
    """A KeyError exception which suggests valid alternatives."""
    def __init__(self, key, alternatives):
        if key is None:
            super().__init__("None: did you mean pokemon.VAL.RANDOM or pokemon.VAL.OPTIMAL?")
        matches = difflib.get_close_matches(key, alternatives)
        matches = ", ".join(matches)
        message = "{0:s}: did you mean {1:s} ?".format(key, matches)
        super().__init__(message)


def diffDict(da, db):
    """Return the differences between da and db."""
    # Return None if they're the same
    if da == db:
        return None
    # add items in da not in db
    rv = {k: v for (k, v) in da.items() if k not in db}
    for k, v in db.items():
        if k not in da:
            rv[k] = v
        elif da[k] != v:
            if isinstance(da[k], dict) and isinstance(v, dict):
                rv[k] = diffDict(da[k], v)
            else:
                rv[k] = [da[k], v]
    return rv


class GameMaster():
    """Class for normalizing and retrieving data from GAME_MASTER.json"""

    _re_type = re.compile(r'POKEMON_TYPE_(.+)')
    _re_pokemon = re.compile(r'V(\d+)_POKEMON_(.+)')
    _re_forms = re.compile(r'FORMS_V(\d+)_POKEMON_(.+)')
    _re_move = re.compile(r'V(\d+)_MOVE_(.+)')
    _re_combat_move = re.compile(r'COMBAT_V(\d+)_MOVE_(.+)')

    K_FAST = "quickMoves"
    K_FAST_ELITE = "eliteQuickMove"
    K_FAST_LEGACY = "legacyQuickMove"
    K_FAST_SUFFIX = "_FAST"  # fast move suffix
    K_CHARGED = "cinematicMoves"
    K_CHARGED_ELITE = "eliteCinematicMove"
    K_CHARGED_LEGACY = "legacyCinematicMove"
    K_SMEARGLE = "SMEARGLE"
    K_SMEARGLE_MOVES = "SMEARGLE_MOVES_SETTINGS"
    K_NORMAL_SUFFIX = "_NORMAL"  # Normal suffix

    K_BONUS_DEF = "defenseBonusMultiplier"

    K_SHADOW_SUFFIX = "_SHADOW"
    K_SHADOW_BONUS_DEF = "shadowPokemonDefenseBonusMultiplier"
    K_SHADOW_BONUS_ATK = "shadowPokemonAttackBonusMultiplier"
    K_PURIFIED_SUFFIX = "_PURIFIED"

    # possible attacks including non Elite TM-able
    _LEGACY = {
        'ACID_FAST': ['GRIMER*', 'KOFFING*', 'WEEZING*', 'MUK*'],
        'ANCIENT_POWER': ['MAMOSWINE*'],
        'BITE_FAST': ['ARCANINE*', 'SHEDINJA*'],
        'BLAST_BURN': ['CHARIZARD*', 'INFERNAPE*', 'TYPHLOSION*'],
        'BLIZZARD': ['CLOYSTER'],
        'BODY_SLAM': ['JIGGLYPUFF*', 'LICKITUNG*', 'LICKILICKY*', 'SLAKING*'],
        'BUG_BITE_FAST': ['BUTTERFREE*', 'BEEDRILL*', 'PARASECT*', 'VENOMOTH*'],
        'BULLDOZE': ['ARCANINE*'],
        'BRINE': [],
        'DARK_PULSE': ['GENGAR*'],
        'DAZZLING_GLEAM': ['ALAKAZAM*'],
        'DRAGON_PULSE': ['AMPHAROS*', 'GYARADOS*', 'LAPRAS*'],
        'DRAGON_TAIL_FAST': ['GYARADOS*'],
        'EARTH_POWER': ['FLYGON*'],
        'EARTHQUAKE': ['GASTRODON*'],
        'EMBER_FAST': ['CHARIZARD*', 'NINETALES*', 'RAPIDASH*', 'MOLTRES*'],
        'FIRE_BLAST': ['PONYTA'],
        'FRENZY_PLANT': ['MEGANIUM*', 'SCEPTILE*', 'TORTERRA*', 'VENUSAUR*'],
        'FRUSTRATION': ['*_SHADOW'],
        'GIGA_DRAIN': [],
        'GRASS_KNOT': ['BRELOOM*'],
        'GUNK_SHOT': ['EKANS*'],
        'HEART_STAMP': [],
        'HYDRO_CANNON': ['EMPOLEON*', 'SWAMPERT*', 'FERALIGATR*'],
        'HYDRO_PUMP_BLASTOISE': ['BLASTOISE*'],
        'HYPER_BEAM': ['MEWTWO'],
        'ICE_BEAM': ['LAPRAS*'],
        'ICE_SHARD_FAST': ['DEWGONG*', 'LAPRAS*'],
        'IRON_HEAD': ['ONIX*'],
        'LAST_RESORT': ['FLAREON*', 'GLACEON*', 'JOLTEON*', 'LEAFEON*', 'UMBREON*', 'VAPOREON*'],
        'LICK_FAST': ['GENGAR*', 'HAUNTER*'],
        'MEGA_DRAIN': [],
        'MEGAHORN': ['RHYDON*'],
        'METEOR_MASH': ['METAGROSS*'],
        'NIGHT_SLASH': ['PERSIAN*'],
        'OMINOUS_WIND': ['GASTLY*'],
        'ORIGIN_PULSE': [],
        'OUTRAGE': ['SALAMENCE*'],
        'PARABOLIC_CHARGE': [],
        'PLAY_ROUGH': ['JIGGLYPUFF*'],
        'POWER_WHIP': ['TANGELA*'],
        'PRECIPICE_BLADES': [],
        'PSYBEAM': ['CHANSEY*'],
        'PSYCHIC': ['ALAKAZAM*', 'GENGAR*'],
        'PSYSHOCK': ['HYPNO*'],
        'PSYSTRIKE': ['MEWTWO', 'MEWTWO_A'],
        'ROCK_SLIDE': ['ONIX*'],
        'THUNDER': ['RAICHU*'],
        'REST': [],
        'RETURN': ['*_PURIFIED'],
        'ROCK_WRECKER': ['RHYPERIOR*'],
        'SACRED_SWORD': ['COBALION*'],
        'SCALD': ['POLIWHIRL*'],
        'SCALD_BLASTOISE': ['BLASTOISE*'],
        'SHADOW_BALL': ['MEWTWO'],
        'SLUDGE_WAVE': ['GENGAR*'],
        'SMACK_DOWN_FAST': ['TYRANITAR*'],
        'SIGNAL_BEAM': ['VOLTORB*'],
        'STOMP': ['HITMONLEE*'],
        'SUCKER_PUNCH_FAST': ['GASTLY*'],
        'SWIFT': ['DODUO'],
        'SUBMISSION': ['POLIWRATH*'],
        'SYNCHRONOISE': ['GALLADE*', 'GARDEVOIR*'],
        'THUNDER_SHOCK_FAST': ['ZAPDOS*'],
        'TWISTER': ['SPEAROW*', 'FEAROW*'],
        'WATER_GUN_FAST': ['FERALIGATR*', 'SEEL*'],
        'WATER_GUN_FAST_BLASTOISE': ['BLASTOISE*'],
        'WEATHER_BALL_ROCK': [],
        'WING_ATTACK_FAST': ['CHARIZARD*'],
        'WRAP_GREEN': [],
        'WRAP_PINK': [],
        'YAWN_FAST': ['SNORLAX'],
        'ZEN_HEADBUTT_FAST': ['EXEGGCUTE'],
    }

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
        self.moves_battle = {}   # moves_battle[move_name] = move_data
        self.moves_combat = {}   # moves_combat[move_name] = move_data
        self._cp_multiplier = None

        """Reconfigure the data to be more useful."""
        self.timestamp = time.ctime(float(self._game_master['timestampMs']) / 1000)

        type_list = []

        items = self._game_master.get('itemTemplate') or self._game_master.get('itemTemplates')

        for item in items:
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

            if tid == "BATTLE_SETTINGS":
                self.settings_battle = item['battleSettings']

            if tid == "COMBAT_SETTINGS":
                self.settings_combat = item['combatSettings']

            if tid == "COMBAT_STAT_STAGE_SETTINGS":
                self.settings_combat_stat_stage = item['combatStatStageSettings']
                self.buff_multiplier_attack = self.settings_combat_stat_stage['attackBuffMultiplier']
                self.buff_multiplier_defense = self.settings_combat_stat_stage['defenseBuffMultiplier']

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
                if 'type1' not in settings:
                    settings['type1'] = settings['type']
                elif 'type' not in settings:
                    settings['type'] = settings['type1']
                self.pokemon[name] = settings
                continue

            # store move settings
            r = self._re_move.match(tid)
            if r:
                settings = item.get('move') or item.get('moveSettings')
                name = settings['movementId']
                assert name not in self.moves_battle
                self.moves_battle[name] = settings
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

        # remove POKEMON_NORMAL if POKEMON is also in the index
        ns_len = -len(GameMaster.K_NORMAL_SUFFIX)
        # make a list of _NORMAL pokemon
        dups = [x for x in self.pokemon.keys() if x.endswith(GameMaster.K_NORMAL_SUFFIX)]
        for name_l in dups:
            name_s = name_l[:ns_len]
            if name_s in self.pokemon:
                diff_p = diffDict(self.pokemon[name], self.pokemon[name])
                if diff_p is not None:
                    # warn if POKEMON and POKEMON_NORMAL are different
                    logging.warning("%s != %s: %s", name_l, name_s, diff_p)
                del self.pokemon[name_l]
            else:
                logging.warning("%s can't find %s", name_l, name_s)

        # add legacy moves to the lists of elite tm-able moves
        legacy_fast = 0
        legacy_charged = 0
        all_moves = set(self.moves_combat.keys())
        all_moves.update(self.moves_battle.keys())
        for move, patterns in GameMaster._LEGACY.items():
            if move not in all_moves:
                raise PokemonKeyError(move, all_moves)
            for pattern in patterns:
                matches = [x for x in self.pokemon.keys() if fnmatch.fnmatch(x, pattern)]
                if not matches:
                    raise Exception("No pokemon match %s:%s", move, pattern)
                for match in matches:
                    if move.endswith(GameMaster.K_FAST_SUFFIX):
                        if move in self.pokemon[match][GameMaster.K_FAST]:
                            # warn if our regular charged table already has this "legacy" move
                            logging.warning("%s already has %s", match, move)
                        elif move in self.pokemon[match].get(GameMaster.K_FAST_ELITE, []):
                            # skip if our Elite TM table already has this legacy move
                            continue
                        else:
                            legacy_fast += 1
                            logging.debug("%d %s legacy %s", legacy_fast, match, move)
                            self.pokemon[match].setdefault(GameMaster.K_FAST_LEGACY, []).append(move)
                    elif move in self.pokemon[match][GameMaster.K_CHARGED]:
                        # warn if our regular charged table already has this "legacy" move
                        logging.warning("%s already has %s", match, move)
                    elif move in self.pokemon[match].get(GameMaster.K_CHARGED_ELITE, []):
                        # skip if our Elite TM table already has this legacy move
                        continue
                    else:
                        legacy_charged += 1
                        logging.debug("%d %s legacy %s", legacy_charged, match, move)
                        self.pokemon[match].setdefault(GameMaster.K_CHARGED_LEGACY, []).append(move)
        logging.info("added %d fast, %d charged legacy moves", legacy_fast, legacy_charged)

        # Add Smeargle's moves
        assert GameMaster.K_FAST not in self.pokemon[GameMaster.K_SMEARGLE]
        assert GameMaster.K_CHARGED not in self.pokemon[GameMaster.K_SMEARGLE]
        self.pokemon[GameMaster.K_SMEARGLE][GameMaster.K_FAST] = smeargle_moves[GameMaster.K_FAST]
        self.pokemon[GameMaster.K_SMEARGLE][GameMaster.K_CHARGED] = smeargle_moves[GameMaster.K_CHARGED]

        # make "type" consistent between moves_combat and moves_battle
        for v in self.moves_battle.values():
            v['type'] = v['pokemonType']

        # re-index effectiveness by enum index
        self.effectiveness = {t.value: self.effectiveness[t.name] for t in [Types(x) for x in TYPE_LIST]}

        # precompute the cp multiplier table for half-levels
        cp_i = self._cp_multiplier[0]
        cpm_table = [cp_i]
        for i in range(1, len(self._cp_multiplier)):
            half_cp = pow(cp_i, 2)
            cp_i = self._cp_multiplier[i]
            half_cp += pow(cp_i, 2)
            half_cp = pow(0.5 * half_cp, 0.5)
            cpm_table.extend([half_cp, cp_i])
        self._cp_multiplier = tuple(cpm_table)

    def cp_multiplier(self, level):
        """Return the CP multiplier for levels [0.0 .. 40.0]"""
        i = round(2 * level) - 2
        return self._cp_multiplier[i]

    def effect(self, attack_type, target_types):
        effect = 1.0
        for ttype in target_types:
            effect *= self.effectiveness[attack_type.value][ttype.value]
        return effect

    def possible_fast(self, name):
        """Return a tuple of fast moves for this pokemon."""
        pdata = self.pokemon[name]
        fast = pdata.get(GameMaster.K_FAST)
        fast_elite = pdata.get(GameMaster.K_FAST_ELITE, [])
        fast_legacy = pdata.get(GameMaster.K_FAST_LEGACY, [])
        return set(fast + fast_elite + fast_legacy)

    def possible_charged(self, name):
        """Return a tuple of charged) moves for this pokemon."""
        pdata = self.pokemon[name]
        charged = pdata.get(GameMaster.K_CHARGED)
        charged_elite = pdata.get(GameMaster.K_CHARGED_ELITE, [])
        charged_legacy = pdata.get(GameMaster.K_CHARGED_LEGACY, [])
        return set(charged + charged_elite + charged_legacy)

    def move_combinations(self, name, r=2):
        """Generate all possible combinations of fast/charged moves for this pokemon."""
        for fast in self.possible_fast(name):
            for charged in itertools.combinations(self.possible_charged(name), r):
                yield fast, charged

    def report(self):
        """Write a summary of the data we've received."""
        logging.info("EFFECTIVENESS %d", len(self.effectiveness))
        logging.info("POKEMON %d", len(self.pokemon))
        logging.info("FORMS %d", len(self.forms))
        logging.info("B MOVES %d", len(self.moves_battle))
        logging.info("C MOVES %d", len(self.moves_combat))


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

    def test_cp_mult(self):
        """Compute correct cp_multiplier values."""
        for level, val in (
                (1.0, 0.094),
                (1.5, 0.1351374318),
                (10.0, 0.4225),
                (10.5, 0.4329264091),
                (20.0, 0.5974),
                (20.5, 0.6048236651),
                (40.0, 0.7903),
                (41.0, 0.79530001)):
            self.assertAlmostEqual(self.gm.cp_multiplier(level), val)


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
