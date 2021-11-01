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
import copy


"""
    Retrieve, parse, and normalize data from GAME_MASTER.json
"""

GAME_MASTER_URL = "https://github.com/PokeMiners/game_masters/blob/master/latest/latest.json?raw=true"
GAME_MASTER_PATH = GAME_MASTER_URL.rsplit('/', 1)[1].rsplit('?', 1)[0]


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
        matches = difflib.get_close_matches(key, alternatives) or alternatives
        matches = ", ".join(sorted(matches))
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


class League():
    """Class representing a Go Battle League."""
    def __init__(self, gm, item):
        self.gm = gm
        self.cp_min = None
        self.cp_max = None
        self.level_max = None

        self.name = item['templateId']
        assert self.name.startswith(GameMaster.K_LEAGUE)
        self.name = self.name[len(GameMaster.K_LEAGUE):]
        data = item['data']['combatLeague']
        self.leagueType = data['leagueType']
        self.pokemonCount = data['pokemonCount']
        self.pokemonWhiteList = set()
        self.withPokemonType = set()

        self.bannedPokemon = set(data.get('bannedPokemon', []))

        for condition in data['pokemonCondition']:
            cpLimit = condition.get('withPokemonCpLimit')
            if cpLimit:
                self.cp_min = cpLimit.get('minCp', self.cp_min)
                self.cp_max = cpLimit.get('maxCp', self.cp_max)
                continue

            levelRange = condition.get('pokemonLevelRange')
            if levelRange:
                self.level_max = levelRange.get('obMaxLevel', self.level_max)
                continue

            pokemonWhiteList = condition.get('pokemonWhiteList')
            if pokemonWhiteList:
                for pattr in pokemonWhiteList['pokemon']:
                    self.pokemonWhiteList.add(pattr.get('form', pattr['id']))
                continue

            pokemonBanList = condition.get('pokemonBanList')
            if pokemonBanList:
                pokemonBanList = pokemonBanList['pokemon']
                pokemonBanList = [x.get('form') or x.get('id') for x in pokemonBanList]
                self.bannedPokemon.update(pokemonBanList)

            withPokemonType = condition.get('withPokemonType')
            if withPokemonType:
                withPokemonType = withPokemonType['pokemonType']
                withPokemonType = {Types[x] for x in withPokemonType}
                self.withPokemonType.update(withPokemonType)

    def __str__(self):
        rv = [self.name]
        if self.cp_min is not None:
            rv.append("min={0:d}".format(self.cp_min))
        if self.cp_max is not None:
            rv.append("max={0:d}".format(self.cp_max))
        if self.level_max is not None:
            rv.append("≤L{0:d}".format(self.level_max))
        if self.bannedPokemon:
            rv.append("({0:d} banned)".format(len(self.bannedPokemon)))
        return " ".join(rv)


class GameMaster():
    """Class for normalizing and retrieving data from GAME_MASTER.json"""

    _re_type = re.compile(r'POKEMON_TYPE_(.+)')
    _re_pokemon = re.compile(r'V(\d+)_POKEMON_(.+)')
    _re_forms = re.compile(r'FORMS_V(\d+)_POKEMON_(.+)')
    _re_move = re.compile(r'V(\d+)_MOVE_(.+)')
    _re_combat_move = re.compile(r'COMBAT_V(\d+)_MOVE_(.+)')

    K_STATS = "stats"
    K_FAST = "quickMoves"
    K_FAST_ELITE = "eliteQuickMove"
    K_FAST_LEGACY = "legacyQuickMove"
    K_FAST_SUFFIX = "_FAST"  # fast move suffix
    K_FAST_INFIX = "_FAST"  # fast move suffix
    K_CHARGED = "cinematicMoves"
    K_CHARGED_ELITE = "eliteCinematicMove"
    K_CHARGED_LEGACY = "legacyCinematicMove"
    K_SMEARGLE = "SMEARGLE"
    K_SMEARGLE_MOVES = "SMEARGLE_MOVES_SETTINGS"
    K_NORMAL_SUFFIX = "_NORMAL"  # Normal suffix
    K_POWER = "power"
    K_ENERGY_DELTA = "energy_delta"
    K_ID_UNIQUE = "pokemonId"
    K_TYPE1 = "type"
    K_TYPE2 = "type2"
    K_IS_MEGA = "is_mega"
    K_EVOLUTION_MEGA = 'TEMP_EVOLUTION_MEGA'
    K_EVOLUTION_OVERRIDES = 'tempEvoOverrides'

    K_BASE_ATTACK = "baseAttack"
    K_BASE_DEFENSE = "baseDefense"
    K_BASE_STAMINA = "baseStamina"

    K_BONUS_DEF = "defenseBonusMultiplier"

    K_LEAGUE = 'COMBAT_LEAGUE_VS_SEEKER_'

    K_SHADOW = "shadow"
    K_SHADOW_SUFFIX = "_SHADOW"
    K_SHADOW_BONUS_DEF = "shadowPokemonDefenseBonusMultiplier"
    K_SHADOW_BONUS_ATK = "shadowPokemonAttackBonusMultiplier"
    K_SHADOW_CHARGED = "shadowChargeMove"
    K_PURIFIED_SUFFIX = "_PURIFIED"
    K_PURIFIED_CHARGED = "purifiedChargeMove"

    # Max level to calculate under normal circumstances
    K_LEVEL_MAX = 51

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
        logging.info("GAME_MASTER loaded in %0.01fms", 1000 * self.startup)

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
        self.leagues = {}        # league info
        self._cp_multiplier = None

        """Reconfigure the data to be more useful."""
        # self.timestamp = self._game_master['timestampMs']
        # self.timestamp = time.ctime(float(self._game_master['timestampMs']) / 1000)

        type_list = []

        for item in self._game_master:
            tid = item['templateId']
            data = item['data']

            # ignore suffixes
            # if self._should_ignore(tid):
            #     continue

            # get cp multiplier
            if tid == "PLAYER_LEVEL_SETTINGS":
                self._cp_multiplier = data['playerLevel']['cpMultiplier']
                continue

            # save smeargle's moves for processing at the end of this loop
            if tid == "SMEARGLE_MOVES_SETTINGS":
                smeargle_moves = data['smeargleMovesSettings']
                continue

            if tid == "BATTLE_SETTINGS":
                self.settings_battle = data['battleSettings']

            if tid == "COMBAT_SETTINGS":
                self.settings_combat = data['combatSettings']

            if tid == "COMBAT_STAT_STAGE_SETTINGS":
                self.settings_combat_stat_stage = data['combatStatStageSettings']
                self.buff_multiplier_attack = self.settings_combat_stat_stage['attackBuffMultiplier']
                self.buff_multiplier_defense = self.settings_combat_stat_stage['defenseBuffMultiplier']

            if tid.startswith(self.K_LEAGUE):
                league = League(self, item)
                assert league.name not in self.leagues
                self.leagues[league.name] = league

            # add forms to self.forms
            r = self._re_forms.match(tid)
            if r:
                settings = data['formSettings']
                baseName = settings['pokemon']
                for formSettings in settings.get('forms', []):
                    formName = formSettings['form']
                    self.forms.setdefault(baseName, []).append(formName)
                continue

            # make type effectiveness
            r = self._re_type.match(tid)
            if r:
                effectiveness = data['typeEffective']
                attack_type = effectiveness['attackType']
                attack_scalar = effectiveness['attackScalar']
                self.effectiveness[attack_type] = attack_scalar
                type_list.append(attack_type)
                continue

            # store pokemon settings
            if tid.endswith('_HOME_FORM_REVERSION') or tid.endswith('_HOME_REVERSION'):
                continue
            r = self._re_pokemon.match(tid)
            if r:
                settings = data['pokemonSettings']
                assert settings
                # form name takes precedence over family name
                name = settings.get('form') or settings.get('uniqueId') or settings.get('pokemonId')
                assert name
                if not settings.get(GameMaster.K_STATS, {}):
                    logging.warning("%s has unspecified stats. Skipping...", name)
                    continue
                settings[GameMaster.K_IS_MEGA] = False
                self.pokemon[name] = settings

                # add the mega evolved forms
                for tempEvo in settings.get(GameMaster.K_EVOLUTION_OVERRIDES, []):
                    evoId = tempEvo.get('tempEvoId')
                    if not evoId.startswith(GameMaster.K_EVOLUTION_MEGA):
                        continue
                    settings_mega = copy.deepcopy(settings)
                    settings_mega[GameMaster.K_TYPE1] = tempEvo['typeOverride1']
                    override_2 = tempEvo.get('typeOverride2')
                    if override_2:
                        settings_mega[GameMaster.K_TYPE2] = override_2
                    settings_mega[GameMaster.K_STATS].update(tempEvo[GameMaster.K_STATS])
                    # new name for the mega including _X or _Y for Charizard
                    name_mega = name + '_MEGA' + evoId[len(GameMaster.K_EVOLUTION_MEGA):]
                    # delete the mega overrides, since megas can't mega-evolve
                    del settings_mega[GameMaster.K_EVOLUTION_OVERRIDES]
                    settings_mega[GameMaster.K_IS_MEGA] = True
                    self.pokemon[name_mega] = settings_mega

                continue

            # store move settings
            r = self._re_move.match(tid)
            if r:
                settings = data['moveSettings']
                name = settings['movementId']
                assert name not in self.moves_battle
                self.moves_battle[name] = settings
                continue

            # store combat (pvp) move settings
            r = self._re_combat_move.match(tid)
            if r:
                settings = data['combatMove']
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
        """Return the CP multiplier for levels [1.0 .. 41.0]"""
        i = round(2 * level) - 2
        return self._cp_multiplier[int(i)]

    def cpm_level(self, cpm):
        """Return the level corresponding to this CP multiplier."""
        diff_last = 9999
        for i, x in enumerate(self._cp_multiplier):
            diff = abs(x - cpm)
            if diff < diff_last:
                diff_last = diff
            else:
                i -= 1
                break
        return (i + 2) / 2

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

    def possible_charged(self, name, is_shadow, is_purified):
        """Return a tuple of charged) moves for this pokemon."""
        pdata = self.pokemon[name]
        charged = pdata.get(GameMaster.K_CHARGED)
        charged_elite = pdata.get(GameMaster.K_CHARGED_ELITE, [])
        charged_legacy = pdata.get(GameMaster.K_CHARGED_LEGACY, [])
        rv = set(charged + charged_elite + charged_legacy)
        if is_shadow:
            rv.add(pdata.get(GameMaster.K_SHADOW, {}).get(GameMaster.K_SHADOW_CHARGED))
        elif is_purified:
            rv.add(pdata.get(GameMaster.K_SHADOW, {}).get(GameMaster.K_PURIFIED_CHARGED))
        return rv

    def best_moves(self, moves):
        """Given a possible move set, eliminate inferior moves that produce less damage and consume more energy."""
        best = {}
        rv = set()
        for move in moves:
            move_data = self.moves_combat[move]
            name = move_data['uniqueId']
            power = move_data.get(GameMaster.K_POWER, 0)
            delta = move_data.get(GameMaster.K_ENERGY_DELTA, 0)
            best_t = best.setdefault(move_data['type'], {})
            add = True
            for bname in list(best_t.keys()):
                bdata = best_t[bname]
                bpower = bdata.get(GameMaster.K_POWER, 0)
                bdelta = bdata.get(GameMaster.K_ENERGY_DELTA, 0)
                # FIXME: consider lasting effects
                if power > bpower and delta > bdelta:
                    # print("Deleting", bname, "<", move_data['uniqueId'])
                    del best_t[bname]
                    rv.remove(bname)
                if power <= bpower and delta <= bdelta:
                    # print("Skipping", bname, ">", move_data['uniqueId'], power, delta)
                    add = False
                    break
            if add:
                # print("Adding", move_data['uniqueId'], power, delta)
                best_t[name] = move_data
                rv.add(name)
        return rv

    def move_combinations(self, name, is_shadow, is_purified, r=2, best=False):
        """Generate all possible combinations of fast/charged moves for this pokemon."""
        possible_fast = self.possible_fast(name)
        possible_charged = self.possible_charged(name, is_shadow, is_purified)
        if best:
            possible_fast = self.best_moves(possible_fast)
            possible_charged = self.best_moves(possible_charged)
        for fast in possible_fast:
            for charged in itertools.combinations(possible_charged, r):
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
            pcharged = self.gm.possible_charged(name, is_shadow=False, is_purified=False)
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

    def test_leagues(self):
        for _, league in sorted(self.gm.leagues.items()):
            print(league)
        print()


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

    unittest.main(failfast=True)
