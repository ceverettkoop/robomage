#include "effect_kind.h"

#include <unordered_map>

EffectKind effect_kind_from_string(const std::string &category) {
    static const std::unordered_map<std::string, EffectKind> table = {
        {"AddMana", EffectKind::AddMana},
        {"GainLife", EffectKind::GainLife},
        {"LoseLife", EffectKind::LoseLife},
        {"Discard", EffectKind::Discard},
        {"Draw", EffectKind::Draw},
        {"ChangeZone", EffectKind::ChangeZone},
        {"ChangeZoneAll", EffectKind::ChangeZoneAll},
        {"RearrangeTopOfLibrary", EffectKind::RearrangeTopOfLibrary},
        {"DealDamage", EffectKind::DealDamage},
        {"PutCounter", EffectKind::PutCounter},
        {"RemoveCounter", EffectKind::RemoveCounter},
        {"ProwessBonus", EffectKind::ProwessBonus},
        {"ExaltedBonus", EffectKind::ExaltedBonus},
        {"Token", EffectKind::Token},
        {"Attach", EffectKind::Attach},
        {"Mill", EffectKind::Mill},
        {"Pump", EffectKind::Pump},
        {"PumpAll", EffectKind::PumpAll},
        {"MultiplyCounter", EffectKind::MultiplyCounter},
        {"ChooseCard", EffectKind::ChooseCard},
        {"Cleanup", EffectKind::Cleanup},
        {"DelayedTrigger", EffectKind::DelayedTrigger},
        {"Untap", EffectKind::Untap},
        {"Destroy", EffectKind::Destroy},
        {"DestroyAll", EffectKind::DestroyAll},
        {"DamageAll", EffectKind::DamageAll},
        {"Counter", EffectKind::Counter},
        {"Surveil", EffectKind::Surveil},
        {"Scry", EffectKind::Scry},
        {"PeekAndReveal", EffectKind::PeekAndReveal},
        // AB$/DB$ Reveal with Random$ True (Urza's Bauble): look at a random card in a
        // player's hand. See effect_reveal.cpp.
        {"Reveal", EffectKind::Reveal},
        {"Phases", EffectKind::Phases},
        {"Dig", EffectKind::Dig},
        {"SylvanLibrary", EffectKind::SylvanLibrary},
        {"WinsGame", EffectKind::WinsGame},
        {"Charm", EffectKind::Charm},
        {"Amass", EffectKind::Amass},
        {"Sacrifice", EffectKind::Sacrifice},
        {"PutCounterAll", EffectKind::PutCounterAll},
        {"SacrificeAll", EffectKind::SacrificeAll},
        {"ImmediateTrigger", EffectKind::ImmediateTrigger},
        {"CopyPermanent", EffectKind::CopyPermanent},
        {"Mobilize", EffectKind::Mobilize},
        {"SacrificeTokens", EffectKind::SacrificeTokens},
        {"RepeatEach", EffectKind::RepeatEach},
        // AB$ Effect that grants "you may cast that card this turn" (Emry, Lurker of the
        // Loch). The transient continuous Effect object is modeled as a per-turn cast
        // permission rather than a stack object; see effect_grant_cast.cpp.
        {"Effect", EffectKind::GrantCast},
        // SP$/DB$ NameCard (Cabal Therapy): the active player names a card; a chained
        // Card.NamedCard sub-ability then references the chosen name (CR 201.4).
        {"NameCard", EffectKind::NameCard},
        // DB$ Animate (Guide of Souls): the targeted permanent "becomes ..." — adds
        // types/subtypes (and, for later cards, base P/T / keywords / creature-ness) for
        // the effect's Duration (Permanent = rest of the game). See effect_animate.cpp.
        {"Animate", EffectKind::Animate},
        // DB$ ChooseNumber (Wrath of the Skies): the resolving controller chooses an integer in
        // [0, Max$]; the pick is stored in cur_game.chosen_number so a chained sub-ability can
        // read it via Count$ChosenNumber. See effect_choose_number.cpp.
        {"ChooseNumber", EffectKind::ChooseNumber},
        // DB$ DigUntil (Amped Raptor): exile cards from the top of the controller's library
        // one at a time until one matches Valid$; revealed cards (and the found one) go to the
        // FoundDestination$/RevealedDestination$ zone; RememberFound$ remembers the found card.
        // See effect_dig_until.cpp.
        {"DigUntil", EffectKind::DigUntil},
        // DB$ Play (Amped Raptor): grant a one-shot permission to cast a Defined$ card (here the
        // remembered nonland) from its current zone (Exile) this turn, paying an alternative
        // RESOURCE cost (PlayCost$ PayEnergy<...> / PayLife<...>) instead of its mana cost. See
        // effect_play.cpp.
        {"Play", EffectKind::Play},
        // DB$/AB$ Earthbend (Badgermole Cub, Ba Sing Se): target land you control becomes a
        // 0/0 creature with haste that's still a land, gets Num$ +1/+1 counters, and is set up
        // with a delayed "when it leaves the battlefield, return it tapped" trigger. The
        // land-animation reuses the Animate extension points on Permanent. See effect_earthbend.cpp.
        {"Earthbend", EffectKind::Earthbend},
        // DB$ Tap (Ba Sing Se's LandTapped replacement SVar): tap Defined$ Self. When ETB$ True
        // it taps the permanent as it enters (the conditional "enters tapped" replacement) — that
        // path is handled by the ENTERS_TAPPED replacement, so this resolve handler covers the
        // generic "tap target/Self" case. See effect_tap.cpp.
        {"Tap", EffectKind::Tap},
        // DB$ RevealHand (Thought-Knot Seer, Duress/Thoughtseize-style): the targeted player
        // (ValidTgts$ Opponent) reveals their hand to all players (CR 701.16). The reveal makes
        // those cards public knowledge so a chained chooser (Chooser$ You ChangeZone) can see and
        // select among them. See effect_reveal_hand.cpp.
        {"RevealHand", EffectKind::RevealHand},
    };
    auto it = table.find(category);
    return (it != table.end()) ? it->second : EffectKind::None;
}
