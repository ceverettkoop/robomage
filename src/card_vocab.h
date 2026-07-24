#ifndef CARD_VOCAB_H
#define CARD_VOCAB_H

#include <string>
#include <unordered_map>

#include "machine_io.h"  // N_CARD_TYPES (embedding vocab size)

// Single source of truth for card name <-> vocab index mapping.
// To add a new card: append an entry to card_vocab_entries.
// N_CARD_TYPES in machine_io.h must be >= (highest index + 1).
struct CardVocabEntry {
    const char *name;
    int index;
};

inline constexpr CardVocabEntry card_vocab_entries[] = {
    {"Mountain", 0}, {"Forest", 1}, {"Lightning Bolt", 2}, {"Grizzly Bears", 3}, {"Volcanic Island", 4},
    {"Scalding Tarn", 5}, {"Flooded Strand", 6}, {"Polluted Delta", 7}, {"Wooded Foothills", 8}, {"Misty Rainforest", 9},
    {"Wasteland", 10}, {"Ponder", 11}, {"Force of Will", 12}, {"Daze", 13}, {"Soul Warden", 14}, {"Tundra", 15},
    {"Delver of Secrets", 16}, {"Insectile Aberration", 17}, {"Flying Men", 18}, {"Island", 19},
    {"Dragon's Rage Channeler", 20}, {"Air Elemental", 21}, {"Counterspell", 22}, {"Lightning Strike", 23},
    {"Brainstorm", 24}, {"Thundering Falls", 25},
    {"Murktide Regent", 26}, {"Mishra's Bauble", 27}, {"Cori-Steel Cutter", 28}, {"Unholy Heat", 29},
    {"Birds of Paradise", 30}, {"Collector Ouphe", 31}, {"Dryad Arbor", 32}, {"Endurance", 33},
    {"Gaea's Cradle", 34}, {"Green Sun's Zenith", 35}, {"Horizon Canopy", 36}, {"Icetill Explorer", 37},
    {"Ignoble Hierarch", 38}, {"Karakas", 39}, {"Keen-Eyed Curator", 40}, {"Knight of the Reliquary", 41},
    {"Noble Hierarch", 42}, {"Once Upon a Time", 43}, {"Plains", 44}, {"Savannah", 45},
    {"Scryb Ranger", 46}, {"Scythecat Cub", 47}, {"Swords to Plowshares", 48}, {"Sylvan Library", 49},
    {"Talon Gates of Madara", 50}, {"Thalia, Guardian of Thraben", 51}, {"Windswept Heath", 52},
    {"Doomsday", 53}, {"Thoughtseize", 54}, {"Dark Ritual", 55}, {"Lotus Petal", 56},
    {"Lion's Eye Diamond", 57}, {"Thassa's Oracle", 58}, {"Personal Tutor", 59}, {"Street Wraith", 60},
    {"Edge of Autumn", 61}, {"Swamp", 62}, {"Undercity Sewers", 63}, {"Underground Sea", 64},
    {"Bloodstained Mire", 65}, {"Verdant Catacombs", 66}, {"Cavern of Souls", 67},
    {"Consider", 68}, {"Duress", 69}, {"Deep Analysis", 70},
    {"Null Rod", 71}, {"Force of Negation", 72}, {"Force of Vigor", 73}, {"Faerie Macabre", 74},
    {"Abrade", 75}, {"Hydroblast", 76}, {"Pyroblast", 77}, {"Knight of Autumn", 78},
    {"Dismember", 79}, {"Meltdown", 80}, {"Deafening Silence", 81}, {"Choke", 82},
    {"Long Goodbye", 83}, {"Fatal Push", 84}, {"Doorkeeper Thrull", 85},
    {"Magus of the Moon", 86},
    {"Barrowgoyf", 87}, {"Dauthi Voidwalker", 88}, {"Stifle", 89},
    {"Life from the Loam", 90},
    {"Disruptor Flute", 91},
    {"Surgical Extraction", 92},
    {"Orcish Bowmasters", 93},
    {"Flow State", 94},
    {"Snuff Out", 95},
    {"Shadowy Backstreet", 96},
    {"Jace, the Mind Sculptor", 97},
    {"Birthing Ritual", 98},
    {"Humility", 99},
    {"Ajani, Nacatl Pariah", 100}, {"Ajani, Nacatl Avenger", 101},
    {"Arid Mesa", 102},
    {"Leyline of the Void", 104},
    {"Ocelot Pride", 105},
    {"Voice of Victory", 107},
    {"Clarion Conqueror", 108},
    {"Goblin Bombardment", 110},
    {"Marsh Flats", 111},
    {"Mindbreak Trap", 112},
    {"Containment Priest", 113},
    {"Plateau", 114},
    {"Price of Progress", 115},
    {"Red Elemental Blast", 116},
    {"Elegant Parlor", 118},
    {"Grafdigger's Cage", 119},
    {"Abundant Countryside", 120},
    {"Aether Vial", 121},
    {"Agate Instigator", 122},
    {"Ancient Tomb", 123},
    {"Arclight Phoenix", 124},
    {"Buried Alive", 126},
    {"Chalice of the Void", 127},
    {"Eldrazi Linebreaker", 129},
    {"Eldrazi Temple", 130},
    {"Emry, Lurker of the Loch", 131},
    {"Erode", 132},
    {"Flagstones of Trokair", 133},
    {"Ghost Quarter", 134},
    {"Glaring Fleshraker", 135},
    {"Kappa Cannoneer", 136},
    {"Kozilek's Command", 137},
    {"Moonshadow", 138},
    {"Seat of the Synod", 140},
    {"Solitude", 141},
    {"Stadium Headliner", 142},
    {"Stoneforge Mystic", 143},
    {"Sunbaked Canyon", 144},
    {"Thoughtcast", 145},
    {"Urza's Bauble", 146},
    {"White Orchid Phantom", 147},
    {"Super Shredder", 148},
    {"Tormod's Crypt", 149},
    {"Bayou", 150},
    {"Metallic Rebuke", 151},
    {"Preordain", 152},
    {"Scrubland", 153},
    {"Aether Spellbomb", 154},
    {"Badlands", 155},
    {"Bojuka Bog", 156},
    {"Recruiter of the Guard", 157},
    {"Wight of the Reliquary", 158},
    {"Cabal Therapy", 159},
    {"Eye of Ugin", 160},
    {"Mai, Scornful Striker", 161},
    {"Natural Order", 162},
    {"Sheoldred's Edict", 163},
    {"Abrupt Decay", 164},
    {"Hymn to Tourach", 165},
    {"Silverbluff Bridge", 166},
    {"Sire of Seven Deaths", 167},
    {"Smash to Smithereens", 168},
    {"Whipflare", 169},
    {"Amped Raptor", 170},
    {"Guide of Souls", 171},
    {"Badgermole Cub", 175},
    {"Mox Opal", 176},
    {"Nethergoyf", 177},
    {"Reality Smasher", 181},
    {"Thought-Knot Seer", 182},
    {"Wrath of the Skies", 186},
    {"Skyclave Apparition", 188},
    {"Ba Sing Se", 197},
    {"Lorehold Charm", 198},
    {"It That Heralds the End", 199},
    {"Petrified Hamlet", 200},
    {"Eiganjo, Seat of the Empire", 201},
    {"Canoptek Scarab Swarm", 202},
    {"Mox Amber", 203},
    {"Alpha Deathclaw", 204},
    {"Wastescape Battlemage", 205},
    {"Consign to Memory", 206},
    {"Forth Eorlingas!", 207},
    {"Phelia, Exuberant Shepherd", 210},
    {"Flickerwisp", 221},
    {"Elvish Reclaimer", 220},
    {"Lavaspur Boots", 224},
    {"Craterhoof Behemoth", 218},
    {"Council's Judgment", 217},
    {"Mystical Dispute", 227},
    {"Pre-War Formalwear", 230},
    {"Meteor Sword", 226},
    {"Otawara, Soaring City", 228},
    {"Pithing Needle", 229},
    {"Prismari Charm", 231},
    {"Shadowspear", 233},
    {"Silent Clearing", 234},
    {"Snow-Covered Island", 235},
    {"Sylvan Safekeeper", 238},
    {"Underground Mortuary", 239},
    {"Wastes", 240},
    {"Toxic Deluge", 241},
    {"Candelabra of Tawnos", 242},
    {"Hide on the Ceiling", 243},
    {"Baleful Strix", 244},
    {"Blue Elemental Blast", 245},
    {"Expedition Map", 246},
    {"Gaddock Teeg", 247},
    {"Grim Monolith", 248},
    {"Stony Silence", 249},
    {"Voltaic Key", 250},
    {"Manifold Key", 251},
    {"Boomerang Basics", 252},
    {"Liquimetal Coating", 253},
    {"Mole Man, Moloid Master", 254},
    {"Mystic Sanctuary", 255},
    {"Pernicious Deed", 256},
    {"Pick Your Poison", 257},
    {"Prismatic Vista", 258},
    {"Price of Freedom", 259},
    {"Witherbloom Command", 260},
    {"Urza's Workshop", 261},
    {"Ugin, Eye of the Storms", 262},
    {"Witch Enchanter", 263},
    {"Witch-Blessed Meadow", 264},
    {"Toxicrene", 265},
    {"Planar Nexus", 266},
    {"The Fantasticar", 267},
    {"Cloak and Dagger, Entwined", 268},
    {"Cabal Ritual", 269},
    {"Trinisphere", 270},
    {"Ensnaring Bridge", 271},
    {"Paradox Engine", 272},
    {"Lorien Revealed", 273},
    {"Urza's Mine", 274},
    {"Urza's Power Plant", 275},
    {"Urza's Tower", 276},
    {"Damping Sphere", 277},
    {"Hexing Squelcher", 278},
    {"Uro, Titan of Nature's Wrath", 279},
    {"Karn, the Great Creator", 280},
    {"Sheltered by Ghosts", 281},
    {"Static Prison", 282},
    {"Veil of Summer", 283},
    {"Into the Flood Maw", 284},
    {"The One Ring", 285},
    {"Lion Sash", 286},
    {"Cityscape Leveler", 287},
    {"Kaito, Bane of Nightmares", 288},
    {"Yorion, Sky Nomad", 289},
    {"Mycosynth Lattice", 290},
    {"Flusterstorm", 291},
    {"Emrakul, the Aeons Torn", 292},
    {"Overlord of the Balemurk", 293},
    {"Outland Liberator", 294}, {"Frenzied Trapbreaker", 295},
    {"Urza's Saga", 296}, {"Summon: Bahamut", 297},
    {"Tamiyo, Inquisitive Student", 298}, {"Tamiyo, Seasoned Scholar", 299},
    {"Tropical Island", 300},
    {"Carpet of Flowers", 301},
    {"Reanimate", 302},
    {"Careful Study", 303},
    {"Griselbrand", 304},
    {"Otherworldly Gaze", 305},
    {"Meticulous Archive", 306},
    {"Spell Pierce", 307},
    {"Tune the Narrative", 308},
    {"Fiery Islet", 309},
    {"Lava Spike", 310},
    {"Exploration", 311},
    {"Crop Rotation", 312},
    {"Rishadan Port", 313},
    {"Skateboard", 314},
    {"Sphere of Resistance", 315},
    {"Boseiju, Who Endures", 316},
    {"Twinshot Sniper", 317},
    {"Yavimaya, Cradle of Growth", 318},
    {"Eidolon of the Great Revel", 319},
    {"Exquisite Firecraft", 320},
    {"Spell Snare", 321},
    {"Goblin Guide", 322},
    {"Archon of Cruelty", 323},
    {"Blast Zone", 324},
    {"Malevolent Rumble", 325},
    {"Searing Blood", 327},
    {"Geist of Saint Traft", 330},
    // Dead//Gone is ONE split card (CR 709) → ONE vocab index (328), aliased under three names so
    // it resolves under both name-normalization schemes. The deck-identity block matches by
    // name_to_uid (lowercases, strips '/'), so "Dead/Gone" → uid "deadgone" lets a "1 Dead/Gone"
    // decklist serialize; the in-game observation matches by ascii_fold (case/punctuation-
    // preserving), so the loaded front name "Dead" and the back-face cast name "Gone" each resolve
    // to 328. "Dead/Gone" is listed LAST so the cost-matrix codegen (last-write-wins per index)
    // leaves an honest zero row for 328: find_card_file's prefix match resolves "Dead"/"Gone" to
    // unrelated dead*/gone* scripts (wrong cost), while "Dead/Gone" finds no file → the standard
    // zero/unresolvable sentinel row (same as DFC backs).
    {"Dead", 328},
    {"Gone", 328},
    {"Dead/Gone", 328},
    {"Fireblast", 329},
    {"Skewer the Critics", 339},
    {"Light Up the Stage", 340},
    {"Roiling Vortex", 341},
    {"Echoing Truth", 331},
    {"Show and Tell", 332},
};

inline constexpr int CARD_VOCAB_SIZE = sizeof(card_vocab_entries) / sizeof(card_vocab_entries[0]);

// Slot N_CARD_TYPES - 1 is reserved as sentinel for all tokens in the ML observation.
static constexpr int TOKEN_SENTINEL = N_CARD_TYPES - 1;

// ── Token identity band ───────────────────────────────────────────────────────
// Vocab indices 900-1022 are reserved for REGISTERED token scripts, so the model can
// tell a Construct from a Clue; TOKEN_SENTINEL (1023) stays the fallback for any token
// whose script is not registered here. Entries are keyed by token SCRIPT filename stem
// (Token::script_name — display names collide across scripts, stems do not). Keep this
// table modest: only token scripts actually created by cards in card_vocab_entries
// (TokenScript$ references plus the engine-synthesized Amass/Investigate/Mobilize stems).
struct TokenVocabEntry {
    const char *script;  // token script stem (matches Token::script_name)
    const char *name;    // display name for observers (card_index_to_name)
    int index;
};

inline constexpr int TOKEN_VOCAB_BASE = 900;  // first index of the token band

inline constexpr TokenVocabEntry token_vocab_entries[] = {
    {"b_0_0_orc_army", "Orc Army", 900},                             // Orcish Bowmasters (Amass)
    {"c_0_0_a_construct_total_artifacts", "Construct", 901},         // Urza's Saga
    {"c_0_1_eldrazi_spawn_sac", "Eldrazi Spawn", 902},               // Glaring Fleshraker, Kozilek's Command
    {"c_1_1_a_insect_flying", "Insect", 903},                        // Canoptek Scarab Swarm
    {"c_1_1_shapeshifter_changeling", "Shapeshifter", 904},          // Abundant Countryside
    {"c_4_4_a_construct_flying_haste", "Construct 4/4", 905},        // The Fantasticar
    {"c_a_clue_draw", "Clue", 906},                                  // Investigate (Tamiyo)
    {"c_a_powerstone", "Powerstone", 907},                           // Cityscape Leveler
    {"moloid", "Moloid", 908},                                       // Mole Man, Moloid Master
    {"r_1_1_warrior", "Warrior", 909},                               // Mobilize (Voice of Victory, Stadium Headliner)
    {"r_2_2_human_knight_trample_haste", "Human Knight", 910},       // Forth Eorlingas!
    {"u_1_1_fish", "Fish", 911},                                     // Into the Flood Maw
    {"u_x_x_illusion", "Illusion", 912},                             // Skyclave Apparition
    {"w_1_1_cat", "Cat", 913},                                       // Ocelot Pride
    {"w_2_1_cat_warrior", "Cat Warrior", 914},                       // Ajani, Nacatl Pariah
    {"w_1_1_monk_prowess", "Monk", 915},                             // Cori-Steel Cutter
    {"w_4_4_angel_flying", "Angel", 916},                            // Geist of Saint Traft
};

inline constexpr int TOKEN_VOCAB_SIZE = sizeof(token_vocab_entries) / sizeof(token_vocab_entries[0]);

// Maps a token script stem to its vocab index in the token band, or -1 when the
// script is unregistered (callers then fall back to TOKEN_SENTINEL).
inline int token_script_to_index(const std::string &script_name) {
    static const std::unordered_map<std::string, int> vocab = [] {
        std::unordered_map<std::string, int> m;
        for (int i = 0; i < TOKEN_VOCAB_SIZE; i++)
            m[token_vocab_entries[i].script] = token_vocab_entries[i].index;
        return m;
    }();
    auto it = vocab.find(script_name);
    return it != vocab.end() ? it->second : -1;
}

// Fold a UTF-8 card name to ASCII so an accented Forge `Name:` (the .txt is
// gitignored and re-fetched accented, e.g. Lorien Revealed's script carries an
// accented "o") still matches an ASCII vocab entry. Transliterates the Latin-1
// Supplement accented letters (UTF-8 lead byte 0xC3) to their base ASCII letter,
// passes ASCII through unchanged, and drops any other non-ASCII byte. This is the
// ONE place name matching crosses the ASCII boundary; name_to_uid (filenames) and
// deck files stay ASCII as before.
inline std::string ascii_fold_card_name(const std::string &s) {
    // index = second UTF-8 byte - 0x80, i.e. codepoint U+00C0..U+00FF; 0 = drop.
    static const char latin1_base[64] = {
        'A', 'A', 'A', 'A', 'A', 'A', 0,   'C', 'E', 'E', 'E', 'E', 'I', 'I', 'I', 'I',  // C0..CF
        'D', 'N', 'O', 'O', 'O', 'O', 'O', 0,   'O', 'U', 'U', 'U', 'U', 'Y', 0,   0,    // D0..DF
        'a', 'a', 'a', 'a', 'a', 'a', 0,   'c', 'e', 'e', 'e', 'e', 'i', 'i', 'i', 'i',  // E0..EF
        'd', 'n', 'o', 'o', 'o', 'o', 'o', 0,   'o', 'u', 'u', 'u', 'u', 'y', 0,   'y',  // F0..FF
    };
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
        } else if (c == 0xC3 && i + 1 < s.size()) {
            unsigned char c2 = static_cast<unsigned char>(s[i + 1]);
            if (c2 >= 0x80 && c2 <= 0xBF) {
                char base = latin1_base[c2 - 0x80];
                if (base) out.push_back(base);
                ++i;  // consumed the continuation byte
            }
        }
        // any other non-ASCII byte is dropped
    }
    return out;
}

// Maps a card name to a 0-based vocabulary index used to encode card identity in
// the machine-mode state vector.  Returns -1 for unregistered cards (encoded as
// the empty/unknown sentinel id).  Matching is ASCII-folded (see above) so an
// accented `CardData::name` resolves to its ASCII vocab entry.
inline int card_name_to_index(const std::string &name) {
    static const std::unordered_map<std::string, int> vocab = [] {
        std::unordered_map<std::string, int> m;
        for (int i = 0; i < CARD_VOCAB_SIZE; i++)
            m[ascii_fold_card_name(card_vocab_entries[i].name)] = card_vocab_entries[i].index;
        return m;
    }();
    auto it = vocab.find(ascii_fold_card_name(name));
    return it != vocab.end() ? it->second : -1;
}

inline const char* card_index_to_name(int idx) {
    static const char** names = [] {
        static const char* table[N_CARD_TYPES]{};
        for (int i = 0; i < N_CARD_TYPES; i++) table[i] = "";
        table[TOKEN_SENTINEL] = "Token";
        for (int i = 0; i < CARD_VOCAB_SIZE; i++)
            table[card_vocab_entries[i].index] = card_vocab_entries[i].name;
        for (int i = 0; i < TOKEN_VOCAB_SIZE; i++)
            table[token_vocab_entries[i].index] = token_vocab_entries[i].name;
        return table;
    }();
    if (idx < 0 || idx >= N_CARD_TYPES) return "???";
    return names[idx];
}

#endif /* CARD_VOCAB_H */
