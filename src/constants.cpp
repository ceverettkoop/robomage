#include <algorithm>
#include <map>
#include <set>
#include <string>

#include "type_constants.h"

static std::string straight_apostrophes(const std::string &name);

const std::set<std::string> all_types = {"Artifact", "Battle", "Conspiracy", "Creature", "Dragon", "Elemental",
    "Enchantment", "Goblin", "Hero", "instant", "Instant", "Jaguar", "Kindred", "Knights", "Land", "Legend",
    "Phenomenon", "Plane", "Planeswalker", "Scheme", "Sorcery", "Stickers", "Summon", "Tribal", "Universewalker",
    "Vanguard", "Wolf"};

// Every subtype of CR 205.3g-q (artifact, enchantment, land, planeswalker, spell, creature,
// planar, dungeon and battle types), spelled as Forge's scripts spell them: a straight apostrophe
// but in Serra’s Realm, Forge's own spelling (subtype_entry reads either apostrophe). Kept from Forge's list though no
// longer in the CR (Un-set and retired types; the registry's positions are by name, so none is removed): "Abian",
// "Alicorn", "Art", "Autobot", "B.O.B.", "Baddest,", "Biggest,", "Brainiac", "Bureaucrat", "Byode", "Cephalid",
// "Chameleon", "Chicken", "Chorus", "Clamfolk", "Cloud", "Cow", "Cyborg", "Deb", "Deer", "Designer", "Donkey", "Duck",
// "Dungeon", "Elemental?", "Elves", "Ersta", "Etiquette", "Fire", "Grandchild", "Gus", "Hatificer", "Head", "Igpay",
// "Inzerva", "Kaladesh", "Key", "Killbot", "Lady", "Mammoth", "Master", "Mime", "Mummy", "Naga", "Nastiest,",
// "Paratrooper", "Penguin", "Pony", "Proper", "Ship", "Spy", "Svega", "The", "Townsfolk", "Vampyre", "Viashino",
// "and/or".
const std::set<std::string> all_subtypes = {"Abian", "Adventure", "Advisor", "Aetherborn", "Ajani", "Alara",
    "Alfava Metraxis", "Alicorn", "Alien", "Ally", "Aminatou", "Amonkhet", "Androzani Minor", "Angel", "Angrath",
    "Antausia", "Antelope", "Apalapucia", "Ape", "Arcane", "Arcavios", "Archer", "Archon", "Arkhos", "Arlinn",
    "Armadillo", "Army", "Art", "Artificer", "Ashiok", "Assassin", "Assembly-Worker", "Astartes", "Atog", "Attraction",
    "Aura", "Aurochs", "Autobot", "Avatar", "Avishkar", "Azgol", "Azra", "B.O.B.", "Background", "Baddest,", "Badger",
    "Bahamut", "Balloon", "Barbarian", "Bard", "Basilisk", "Basri", "Bat", "Bear", "Beast", "Beaver", "Beeble",
    "Beholder", "Belenon", "Berserker", "Biggest,", "Bird", "Bison", "Blinkmoth", "Blood", "Boar", "Bobblehead",
    "Bolas", "Bolas's Meditation Realm", "Book", "Brainiac", "Bringer", "Brushwagg", "Bureaucrat", "Byode", "C'tan",
    "Calix", "Camarid", "Camel", "Capenna", "Capybara", "Caribou", "Carrier", "Cartouche", "Case", "Cat", "Cave",
    "Centaur", "Cephalid", "Chameleon", "Chandra", "Chicken", "Child", "Chimera", "Chorus", "Citizen", "Clamfolk",
    "Class", "Cleric", "Cloud", "Clown", "Clue", "Cockatrice", "Comet", "Construct", "Contraption", "Cow", "Coward",
    "Coyote", "Crab", "Cridhe", "Crocodile", "Curse", "Custodes", "Cyberman", "Cyborg", "Cyclops", "Dack", "Dakkon",
    "Dalek", "Daretti", "Darillium", "Dauthi", "Davriel", "Deb", "Deer", "Dellian", "Demigod", "Demon", "Desert",
    "Deserter", "Designer", "Detective", "Devil", "Dihada", "Dinosaur", "Djinn", "Doctor", "Dog", "Dominaria", "Domri",
    "Donkey", "Dovin", "Dragon", "Drake", "Dreadnought", "Drix", "Drone", "Druid", "Dryad", "Duck", "Dungeon", "Dwarf",
    "Earth", "Echidna", "Echoir", "Efreet", "Egg", "Elder", "Eldraine", "Eldrazi", "Elemental", "Elemental?",
    "Elephant", "Elf", "Elk", "Ellywick", "Elminster", "Elspeth", "Elves", "Employee", "Equilor", "Equipment",
    "Ergamon", "Ersta", "Estrid", "Etiquette", "Eye", "Fabacin", "Faerie", "Ferret", "Fiora", "Fire", "Fish",
    "Flagbearer", "Food", "Forest", "Fortification", "Fox", "Fractal", "Freyalise", "Frog", "Fungus", "Gallifrey",
    "Gamer", "Gargantikar", "Gargoyle", "Garruk", "Gate", "Germ", "Giant", "Gideon", "Giraffe", "Gith", "Glimmer",
    "Gnoll", "Gnome", "Goat", "Gobakhan", "Goblin", "God", "Gold", "Golem", "Gorgon", "Grandchild", "Graveborn",
    "Gremlin", "Griffin", "Grist", "Guest", "Guff", "Gus", "Hag", "Halfling", "Hamster", "Harpy", "Hatificer", "Head",
    "Hedgehog", "Hellion", "Hero", "Hippo", "Hippogriff", "Homarid", "Homunculus", "Horror", "Horse",
    "Horsehead Nebula", "Huatli", "Human", "Hydra", "Hyena", "Igpay", "Ikoria", "Illusion", "Imp", "Incarnation",
    "Incubator", "Infinity", "Inkling", "Innistrad", "Inquisitor", "Insect", "Inzerva", "Iquatana", "Ir", "Island",
    "Ixalan", "Jace", "Jackal", "Jared", "Jaya", "Jellyfish", "Jeska", "Juggernaut", "Junk", "Kaito", "Kaladesh",
    "Kaldheim", "Kamigawa", "Kandoka", "Kangaroo", "Karn", "Karsus", "Kasmina", "Kavu", "Kaya", "Kephalai", "Key",
    "Killbot", "Kinshala", "Kiora", "Kirin", "Kithkin", "Knight", "Kobold", "Kolbahan", "Kor", "Koth", "Kraken",
    "Kylem", "Kyneth", "Lady", "Lair", "Lamia", "Lammasu", "Lander", "Leech", "Lemur", "Lesson", "Leviathan",
    "Lhurgoyf", "Licid", "Liliana", "Lizard", "Llama", "Lobster", "Locus", "Lolth", "Lorwyn", "Lukka", "Luvion",
    "Mammoth", "Manticore", "Map", "Mars", "Master", "Masticore", "Mercadia", "Mercenary", "Merfolk", "Metathran",
    "Mime", "Mine", "Minion", "Minotaur", "Minsc", "Mirrodin", "Mite", "Moag", "Mole", "Monger", "Mongoose", "Mongseng",
    "Monk", "Monkey", "Moogle", "Moon", "Moonfolk", "Mordenkainen", "Mount", "Mountain", "Mouse", "Mummy", "Muraganda",
    "Mutagen", "Mutant", "Myr", "Mystic", "Naga", "Nahiri", "Narset", "Nastiest,", "Nautilus", "Necron", "Necros",
    "Nephilim", "New Earth", "New Phyrexia", "Nightmare", "Nightstalker", "Niko", "Ninja", "Nissa", "Nixilis", "Noble",
    "Noggle", "Nomad", "Nymph", "Octopus", "Ogre", "Oko", "Omen", "Ooze", "Orb", "Orc", "Orgg", "Otter", "Ouphe",
    "Outside Mutter's Spiral", "Ox", "Oyster", "Pangolin", "Paratrooper", "Peasant", "Pegasus", "Penguin", "Pentavite",
    "Performer", "Pest", "Phelddagrif", "Phoenix", "Phyrexia", "Phyrexian", "Pilot", "Pincher", "Pirate", "Plains",
    "Planet", "Plant", "Platypus", "Pony", "Porcupine", "Possum", "Power-Plant", "Powerstone", "Praetor", "Primarch",
    "Prism", "Processor", "Proper", "Pyrulea", "Qu", "Quintorius", "Rabbit", "Rabiah", "Raccoon", "Ral", "Ranger",
    "Rat", "Rath", "Ravnica", "Rebel", "Reflection", "Regatha", "Rhino", "Rigger", "Robot", "Rogue", "Role", "Room",
    "Rowan", "Rune", "Sable", "Saga", "Saheeli", "Salamander", "Samurai", "Samut", "Sand", "Saproling", "Sarkhan",
    "Satyr", "Scarecrow", "Scientist", "Scion", "Scorpion", "Scout", "Sculpture", "Seal", "Segovia", "Serf", "Serpent",
    "Serra", "Serra’s Realm", "Servo", "Shade", "Shadowmoor", "Shaman", "Shandalar", "Shapeshifter", "Shard", "Shark",
    "Sheep", "Shenmeng", "Ship", "Shrine", "Siege", "Siren", "Sivitri", "Skaro", "Skeleton", "Skunk", "Slith", "Sliver",
    "Sloth", "Slug", "Snail", "Snake", "Soldier", "Soltari", "Sorcerer", "Sorin", "Spacecraft", "Spawn", "Specter",
    "Spellshaper", "Sphere", "Sphinx", "Spider", "Spike", "Spirit", "Splinter", "Sponge", "Spy", "Squid", "Squirrel",
    "Starfish", "Stone", "Surrakar", "Survivor", "Svega", "Swamp", "Symbiote", "Synth", "Szat", "Tamiyo", "Tarkir",
    "Tasha", "Teferi", "Tentacle", "Tetravite", "Teyo", "Tezzeret", "Thalakos", "The", "The Abyss", "The Dalek Asylum",
    "The Library", "Theros", "Thopter", "Thrull", "Tibalt", "Tiefling", "Time", "Time Lord", "Tower", "Town",
    "Townsfolk", "Toy", "Trap", "Treasure", "Treefolk", "Trenzalore", "Trilobite", "Triskelavite", "Troll", "Turtle",
    "Tyranid", "Tyvar", "Ugin", "Ulgrotha", "Undercity", "Unicorn", "Unknown Planet", "Urza", "Urza's", "Utrom",
    "Valla", "Vampire", "Vampyre", "Varmint", "Vedalken", "Vehicle", "Venser", "Viashino", "Villain", "Vivien",
    "Volver", "Vraska", "Vronos", "Vryn", "Wall", "Walrus", "Warlock", "Warrior", "Weasel", "Weird", "Werewolf",
    "Whale", "Wildfire", "Will", "Windgrace", "Wizard", "Wolf", "Wolverine", "Wombat", "Worm", "Wraith", "Wrenn",
    "Wurm", "Xenagos", "Xerex", "Yanggu", "Yanling", "Yeti", "Zariel", "Zendikar", "Zhalfir", "Zombie", "Zubera",
    "and/or"};

const std::set<std::string> all_supertypes = {"Basic", "Host", "Legendary", "Ongoing", "Snow", "World"};

// A name with the rules' curly apostrophe (U+2019) spelled straight, as the scripts spell it.
static std::string straight_apostrophes(const std::string &name) {
    std::string out;
    for (size_t i = 0; i < name.size(); ++i) {
        if (name.compare(i, 3, "\xE2\x80\x99") == 0) {
            out += '\'';
            i += 2;
        } else {
            out += name[i];
        }
    }
    return out;
}

const std::string *subtype_entry(const std::string &name) {
    static const std::map<std::string, const std::string *> by_spelling = [] {
        std::map<std::string, const std::string *> m;
        for (const std::string &s : all_subtypes) m.emplace(straight_apostrophes(s), &s);
        return m;
    }();
    auto it = by_spelling.find(straight_apostrophes(name));
    return it == by_spelling.end() ? nullptr : it->second;
}

size_t subtype_max_words() {
    static const size_t most = [] {
        size_t n = 1;
        for (const std::string &s : all_subtypes)
            n = std::max(n, static_cast<size_t>(std::count(s.begin(), s.end(), ' ')) + 1);
        return n;
    }();
    return most;
}
