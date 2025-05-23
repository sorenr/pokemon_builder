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

GAME_MASTER_URL = "https://raw.githubusercontent.com/PokeMiners/game_masters/master/latest/latest.json"
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

    def __int__(self):
        return self.value


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
        self.allowTempEvos = data.get('allowTempEvos')

        self.bannedPokemon = set(data.get('bannedPokemon', []))

        for condition in data['pokemonCondition']:
            cpLimit = condition.get('withPokemonCpLimit')
            if cpLimit:
                self.cp_min = cpLimit.get('minCp', self.cp_min)
                self.cp_max = cpLimit.get('maxCp', self.cp_max)
                continue

            levelRange = condition.get('pokemonLevelRange')
            if levelRange:
                self.level_max = levelRange['maxLevel']
                continue

            pokemonWhiteList = condition.get('pokemonWhiteList')
            if pokemonWhiteList:
                for pattr in pokemonWhiteList['pokemon']:
                    pattr = pattr.get(GameMaster.K_FORMS) or [pattr['id']]
                    self.pokemonWhiteList.update( pattr )
                continue

            pokemonBanList = condition.get('pokemonBanList')
            if pokemonBanList:
                for bannedPokemon in pokemonBanList['pokemon']:
                    bannedPokemon = bannedPokemon.get(GameMaster.K_FORMS) or [bannedPokemon.get('id')]
                    self.bannedPokemon.update(bannedPokemon)
                continue

            withPokemonType = condition.get('withPokemonType')
            if withPokemonType:
                withPokemonType = withPokemonType['pokemonType']
                withPokemonType = {Types[x] for x in withPokemonType}
                self.withPokemonType.update(withPokemonType)
                continue

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

    K_BASE_ATTACK = "baseAttack"
    K_BASE_DEFENSE = "baseDefense"
    K_BASE_STAMINA = "baseStamina"
    K_BONUS_DEF = "defenseBonusMultiplier"
    K_CHARGED = "cinematicMoves"
    K_CHARGED_ELITE = "eliteCinematicMove"
    K_CHARGED_LEGACY = "legacyCinematicMove"
    K_ENERGY_DELTA = "energyDelta"
    K_EVOLUTION_MEGA = 'TEMP_EVOLUTION_MEGA'
    K_EVOLUTION_OVERRIDES = 'tempEvoOverrides'
    K_FAST = "quickMoves"
    K_FAST_ELITE = "eliteQuickMove"
    K_FAST_LEGACY = "legacyQuickMove"
    K_FAST_SUFFIX = "_FAST"  # fast move suffix
    K_FAST_INFIX = "_FAST"  # fast move suffix
    K_FORM = "form"
    K_FORMS = "forms"
    K_ID_UNIQUE = "pokemonId"
    K_IS_MEGA = "is_mega"
    K_LEAGUE = 'COMBAT_LEAGUE_VS_SEEKER_'
    K_MOVE_NAME = 'uniqueId'
    K_MOVE_TYPE = 'type'
    K_MOVE_NUM = 'moveNum'
    K_NORMAL_SUFFIX = "_NORMAL"  # Normal suffix
    K_POWER = "power"
    K_PURIFIED_SUFFIX = "_PURIFIED"
    K_PURIFIED_CHARGED = "purifiedChargeMove"
    K_SHADOW = "shadow"
    K_SHADOW_SUFFIX = "_SHADOW"
    K_SHADOW_BONUS_DEF = "shadowPokemonDefenseBonusMultiplier"
    K_SHADOW_BONUS_ATK = "shadowPokemonAttackBonusMultiplier"
    K_SHADOW_CHARGED = "shadowChargeMove"
    K_SMEARGLE = "SMEARGLE"
    K_SMEARGLE_MOVES = "SMEARGLE_MOVES_SETTINGS"
    K_STATS = "stats"
    K_TYPE1 = "type"
    K_TYPE2 = "type2"

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
    
    def smeargle_simplify(self):
        if GameMaster.K_SMEARGLE in self.pokemon:
            self.pokemon[GameMaster.K_SMEARGLE][GameMaster.K_FAST] = ['LOCK_ON_FAST']
            self.pokemon[GameMaster.K_SMEARGLE][GameMaster.K_CHARGED] = ['FLYING_PRESS']

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
            smeargle_moves = None

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
                for formSettings in filter(None, settings.get('forms', [])):
                    formName = formSettings['form']
                    self.forms.setdefault(baseName, set()).add(formName)
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

                name = settings.get(self.K_FORM) or settings.get('uniqueId') or settings.get('pokemonId')
                assert name

                # One form of FORMS_V0025_POKEMON_PIKACHU has an integer name
                name = str(name)

                if not settings.get(GameMaster.K_STATS, {}):
                    logging.warning("%s has unspecified stats. Skipping...", name)
                    continue
                settings[GameMaster.K_IS_MEGA] = False
                self.pokemon[name] = settings

                # add the mega evolved forms
                for tempEvo in settings.get(GameMaster.K_EVOLUTION_OVERRIDES, []):
                    evoId = tempEvo.get('tempEvoId')
                    if not evoId or not evoId.startswith(GameMaster.K_EVOLUTION_MEGA):
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

            settings_move = None
            settings_move_context = None

            # store move settings
            r = self._re_move.match(tid)
            if r:
                settings_move = data['moveSettings']
                settings_move_context = self.moves_battle
            else:
                r = self._re_combat_move.match(tid)
                if r:
                    settings_move = data['combatMove']
                    settings_move_context = self.moves_combat

            if settings_move:
                moveNum = int(r.group(1))
                moveName = r.group(2)
                assert moveNum not in settings_move_context
                assert moveName not in settings_move_context
                settings_move[self.K_MOVE_NUM] = moveNum
                settings_move[self.K_MOVE_NAME] = moveName
                settings_move_context[moveNum] = settings_move
                settings_move_context[moveName] = settings_move
                continue

            # warn about a possibly important item we're ignoring
            # logging.debug("Won't process %s", tid)

        # Add Smeargle's moves
        assert GameMaster.K_FAST not in self.pokemon[GameMaster.K_SMEARGLE]
        assert GameMaster.K_CHARGED not in self.pokemon[GameMaster.K_SMEARGLE]
        if smeargle_moves:
            self.pokemon[GameMaster.K_SMEARGLE][GameMaster.K_FAST] = smeargle_moves[GameMaster.K_FAST]
            self.pokemon[GameMaster.K_SMEARGLE][GameMaster.K_CHARGED] = smeargle_moves[GameMaster.K_CHARGED]
        else:
            print("No smeargle moves. Deleting Smeargle.")
            del self.pokemon[GameMaster.K_SMEARGLE]

        # Add non-tm moves
        self.pokemon['RAYQUAZA'][GameMaster.K_CHARGED].append('DRAGON_ASCENT')
        self.pokemon['PALKIA_ORIGIN'][GameMaster.K_CHARGED].append('SPACIAL_REND')
        self.pokemon['DIALGA_ORIGIN'][GameMaster.K_CHARGED].append('ROAR_OF_TIME')
        self.pokemon['NECROZMA_DAWN_WINGS'][GameMaster.K_CHARGED].append('MOONGEIST_BEAM')
        self.pokemon['NECROZMA_DUSK_MANE'][GameMaster.K_CHARGED].append('SUNSTEEL_STRIKE')
        self.pokemon['KYUREM_WHITE'][GameMaster.K_CHARGED].append('ICE_BURN')
        self.pokemon['KYUREM_BLACK'][GameMaster.K_CHARGED].append('FREEZE_SHOCK')

        # make "type" consistent between moves_combat and moves_battle
        for v in self.moves_battle.values():
            v['type'] = v['pokemonType']

        # Some moves in pokemon move lists are numeric. Change them to strings.
        # Example: Xerneas 387 => GEOMANCY_FAST
        for poke, pdata in self.pokemon.items():
            for moves_type in [self.K_FAST, self.K_FAST_ELITE, self.K_FAST_LEGACY, self.K_CHARGED, self.K_CHARGED_ELITE, self.K_CHARGED_LEGACY]:
                for i, move_old in enumerate(pdata.get(moves_type, [])):
                    print( moves_type, i, move_old )
                    move_new = self.moves_combat[move_old]
                    move_new = move_new[self.K_MOVE_NAME]
                    if type(move_old) is not type(move_new):
                        print(poke, moves_type, move_old, "=>", move_new)
                        pdata[moves_type][i] = move_new

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
            move_name = move_data[GameMaster.K_MOVE_NAME]
            move_power = move_data.get(GameMaster.K_POWER, 0)
            move_delta = move_data.get(GameMaster.K_ENERGY_DELTA, 0)
            move_type = move_data[GameMaster.K_MOVE_TYPE]
            moves_type_best = best.setdefault(move_type, set())
            print("CHECKING", move_name, move_type, move_power, move_delta)
            print(move_data)
            add = True
            for move_best_name in list(moves_type_best):
                move_best_data = self.moves_combat[move_best_name]
                move_best_power = move_best_data.get(GameMaster.K_POWER, 0)
                move_best_delta = move_best_data.get(GameMaster.K_ENERGY_DELTA, 0)
                # FIXME: consider lasting effects
                if move_power > move_best_power and move_delta > move_best_delta:
                    print("Deleting", move_best_name, move_best_power, move_best_delta, "<", move_name, move_power, move_delta)
                    moves_type_best.remove(move_best_name)
                if move_power <= move_best_power and move_delta <= move_best_delta:
                    print("Skipping", move_best_name, move_best_power, move_best_delta, ">", move_name, move_power, move_delta)
                    add = False
                    break
            if add:
                print("Adding", move_name, move_power, move_delta)
                moves_type_best.add(move_name)
        import pprint
        pprint.pprint(best)
        for _, best_moves in best.items():
            rv = rv.union(best_moves)
        return rv

    def move_combinations(self, name, is_shadow, is_purified, r=2, best=False):
        """Generate all possible combinations of fast/charged moves for this pokemon."""
        possible_fast = self.possible_fast(name)
        possible_charged = self.possible_charged(name, is_shadow, is_purified)
        if best:
            possible_fast = self.best_moves(possible_fast)
            possible_charged = self.best_moves(possible_charged)
        for fast in possible_fast:
            if len(possible_charged) <= r:
                yield fast, tuple(possible_charged)
            else:
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
