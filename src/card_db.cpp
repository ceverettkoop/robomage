#include "card_db.h"

const Card lightning_bolt = {
    "Lightning Bolt", 
    "Deals three damage to any target",
    0
};

const Card mountain = {
    "Mountain", 
    "{T} Add {R} to your mana pool",
    1
};

std::unordered_map<uint32_t, Card> card_db = {
    {0, lightning_bolt},
    {1, mountain}
};