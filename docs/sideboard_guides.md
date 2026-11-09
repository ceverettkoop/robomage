# League sideboard guides

Sideboarding for the `scripted:hard+sb` teacher tier: for each guided league deck, what it brings
in and cuts against each other guided deck, and what it changes on the draw. A prior for the warm
start, not the target: the trained policy is meant to learn sideboarding beyond this table.

Not yet reviewed: the death_and_taxes, lands_top8, ur_tempo and maverick sections and uw_control's
row against maverick.

The tier reads the machine-readable copy `python/sideboard_guides.json` (draw lands, every table
row and the held-out matchups, card names as the league deck files spell them);
`python/pycheck.py` (the gate's `smoke` tier) checks it against the tables here and the deck files.

## How the guides are applied

- A guide is a **target configuration** per (own deck, opponent deck, on the play or the draw). At
  every sideboarding phase the teacher swaps from its current 75 to that target, so game 3 starts
  from game 2's configuration and the draw-land swap is undone or applied as play/draw flips.
- The opponent's deck is known from its registered list (open decklists). An opponent that is not a
  guided deck, a matchup with no row, or a held-out matchup (below) gets Done (no swaps). A league
  deck with no guide keeps its maindeck, and its opponents keep theirs against it.
- **On the play:** the matchup's *In* and *Out*.
- **On the draw:** also the deck's draw land out and the matchup's *Draw +1* card in. lands_top8,
  maverick and wubg_doomsday cut no land: their guides are the same on the play and the draw.
- Every swap is one card in and one out (the engine's balance rule), so *In* and *Out* have equal
  counts, and every phase stays under the engine's 15-swap cap.

## Draw land per deck

| Deck | Cut on the draw |
|---|---|
| bug | 1 Swamp |
| burn | 1 Mountain |
| death_and_taxes | 1 Plains |
| lands_top8 | none |
| maverick | none |
| reanimator | 1 Swamp |
| ur_tempo | 1 Thundering Falls |
| uw_control | 1 Island |
| wrb_energy | 1 Plains |
| wubg_doomsday | none |

## bug

Sideboard: 2 Carpet of Flowers, 2 Consign to Memory, 2 Endurance, 1 Force of Vigor, 1 Hydroblast,
1 Null Rod, 2 Pernicious Deed, 1 Force of Negation, 1 Surgical Extraction, 1 Toxic Deluge,
1 Veil of Summer.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| burn | 1 Hydroblast, 2 Endurance | 2 Thoughtseize, 1 Life from the Loam | Pernicious Deed |
| death_and_taxes | 2 Pernicious Deed, 1 Toxic Deluge | 2 Thoughtseize, 1 Lorien Revealed | Force of Vigor |
| lands_top8 | 1 Force of Vigor, 1 Surgical Extraction, 2 Consign to Memory, 1 Pernicious Deed | 2 Orcish Bowmasters, 2 Fatal Push, 1 Witherbloom Command | Null Rod |
| reanimator | 1 Surgical Extraction, 2 Consign to Memory, 1 Force of Negation, 2 Endurance | 3 Fatal Push, 2 Witherbloom Command, 1 Lorien Revealed | Veil of Summer |
| ur_tempo | 1 Hydroblast, 1 Null Rod, 1 Toxic Deluge | 2 Thoughtseize, 1 Life from the Loam | Veil of Summer |
| uw_control | 2 Carpet of Flowers, 1 Veil of Summer, 1 Force of Negation, 1 Consign to Memory | 2 Fatal Push, 2 Witherbloom Command, 1 Sheoldred's Edict | Force of Vigor |
| wrb_energy | 2 Pernicious Deed, 1 Toxic Deluge, 1 Hydroblast | 2 Thoughtseize, 1 Lorien Revealed, 1 Life from the Loam | Endurance |
| wubg_doomsday | 2 Consign to Memory, 1 Surgical Extraction, 1 Veil of Summer, 1 Force of Negation | 3 Fatal Push, 2 Sheoldred's Edict | Carpet of Flowers |

Notes: Consign counters Thassa's Oracle's trigger, Archon's and Atraxa's enter triggers, Saga
chapters and colorless spells (The One Ring, Mox Diamond, Sphere). Edicts stay against reanimator
and lands (Marit Lage).

## burn

Sideboard: 2 Dead // Gone, 4 Leyline of the Void, 2 Roiling Vortex, 3 Searing Blood,
4 Smash to Smithereens.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Roiling Vortex, 2 Searing Blood | 1 Light Up the Stage, 3 Skewer the Critics | Dead // Gone |
| death_and_taxes | 3 Searing Blood, 2 Dead // Gone, 2 Smash to Smithereens | 4 Eidolon of the Great Revel, 2 Lava Spike, 1 Light Up the Stage | Smash to Smithereens |
| lands_top8 | 4 Leyline of the Void, 2 Smash to Smithereens | 2 Exquisite Firecraft, 1 Light Up the Stage, 3 Skewer the Critics | Smash to Smithereens |
| reanimator | 4 Leyline of the Void, 2 Roiling Vortex | 2 Exquisite Firecraft, 1 Light Up the Stage, 3 Skewer the Critics | Dead // Gone |
| ur_tempo | 3 Searing Blood, 2 Roiling Vortex | 1 Light Up the Stage, 3 Skewer the Critics, 1 Rift Bolt | Smash to Smithereens |
| uw_control | 2 Roiling Vortex, 2 Smash to Smithereens | 1 Light Up the Stage, 3 Skewer the Critics | Searing Blood |
| wrb_energy | 3 Searing Blood, 1 Dead // Gone, 2 Roiling Vortex | 4 Eidolon of the Great Revel, 1 Lava Spike, 1 Light Up the Stage | Dead // Gone |
| wubg_doomsday | 2 Roiling Vortex | 1 Light Up the Stage, 1 Skewer the Critics | Dead // Gone |

Notes: Roiling Vortex punishes free spells (Force of Will, Daze, Lotus Petal, Lion's Eye Diamond)
and stops lifegain (Uro, Guide of Souls, Ocelot Pride). Smash hits The One Ring, Cori-Steel Cutter,
the Stoneforge package and Vial. Gone answers a reanimated Atraxa or Griselbrand.

## death_and_taxes

Sideboard: 1 Containment Priest, 1 Damping Sphere, 3 Deafening Silence, 2 Erode, 4 Leyline of the
Void, 3 Wrath of the Skies, 1 Yorion, Sky Nomad (the companion: never boarded in).

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Erode | 1 Meteor Sword, 1 Disruptor Flute | Deafening Silence |
| burn | 2 Erode | 1 Meteor Sword, 1 Canoptek Scarab Swarm | Containment Priest |
| lands_top8 | 4 Leyline of the Void | 3 Skyclave Apparition, 1 Meteor Sword | Deafening Silence |
| reanimator | 4 Leyline of the Void, 1 Containment Priest | 1 Pre-War Formalwear, 1 Meteor Sword, 1 Flickerwisp, 2 Stoneforge Mystic | Deafening Silence |
| ur_tempo | 2 Erode, 1 Deafening Silence | 1 Meteor Sword, 1 Disruptor Flute, 1 Canoptek Scarab Swarm | Deafening Silence |
| uw_control | 2 Deafening Silence, 2 Erode | 1 Meteor Sword, 1 Disruptor Flute, 2 Swords to Plowshares | Deafening Silence |
| wrb_energy | 2 Erode | 1 Meteor Sword, 1 Canoptek Scarab Swarm | Containment Priest |
| wubg_doomsday | 3 Deafening Silence, 1 Damping Sphere | 3 Swords to Plowshares, 1 Meteor Sword | Erode |

Notes: Containment Priest also exiles the deck's own blinked and Formalwear-returned creatures, so
against reanimator it comes in with the Formalwear and Flickerwisp cuts, and the Stoneforges go
out with both Equipment. Disruptor Flute is main: it names Reanimate (reanimator), Doomsday,
Thespian's Stage (lands), Lightning Bolt (burn), Goblin Bombardment (energy); which name to pick
is the teacher's choice at the name prompt, and one copy goes where it names little (bug, ur_tempo,
uw_control). Canoptek Scarab Swarm stays against the graveyard decks (bug, lands, reanimator, doomsday).
Karakas stays in every matchup (Atraxa, Griselbrand, Marit Lage, Uro).

## lands_top8

Sideboard: 2 Choke, 2 Damping Sphere, 2 Dismember, 3 Disruptor Flute, 1 Drop of Honey, 2 Endurance,
3 Force of Vigor.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Endurance, 1 Disruptor Flute | 1 Skateboard, 1 Expedition Map, 1 Malevolent Rumble | - |
| burn | 2 Endurance, 1 Drop of Honey, 2 Disruptor Flute | 1 Pithing Needle, 1 Expedition Map, 1 Skateboard, 2 Crop Rotation | - |
| death_and_taxes | 3 Force of Vigor, 1 Disruptor Flute, 2 Dismember | 2 Sphere of Resistance, 1 Soul-Guide Lantern, 1 Skateboard, 1 Expedition Map, 1 Malevolent Rumble | - |
| reanimator | 2 Dismember, 1 Drop of Honey, 2 Disruptor Flute | 1 Skateboard, 1 Expedition Map, 1 Pithing Needle, 2 Malevolent Rumble | - |
| ur_tempo | 2 Dismember, 1 Drop of Honey, 2 Choke | 1 Skateboard, 1 Expedition Map, 1 Soul-Guide Lantern, 2 Malevolent Rumble | - |
| uw_control | 1 Disruptor Flute, 3 Force of Vigor, 1 Damping Sphere | 1 Soul-Guide Lantern, 1 Skateboard, 2 Malevolent Rumble, 1 Expedition Map | - |
| wrb_energy | 2 Dismember, 1 Drop of Honey, 1 Endurance, 1 Force of Vigor | 1 Soul-Guide Lantern, 1 Skateboard, 1 Expedition Map, 2 Sphere of Resistance | - |
| wubg_doomsday | 1 Damping Sphere, 1 Disruptor Flute, 1 Force of Vigor | 1 Maze of Ith, 1 Soul-Guide Lantern, 1 Skateboard | - |

Notes: Disruptor Flute does a second Pithing Needle's work: it names Wasteland or Karakas (DnT),
The One Ring or Teferi (uw_control), Kaito (bug), Lion's Eye Diamond (doomsday), and taxes a spell
besides, Lightning Bolt (burn) or Reanimate (reanimator). Dismember and Drop of Honey are the
creature answers, Drop of Honey also a lone reanimated creature. Soul-Guide Lantern stays against
bug (Uro's escape), burn and reanimator.

## maverick

Sideboard: 2 Choke, 1 Disruptor Flute, 3 Doorkeeper Thrull, 2 Erode, 2 Force of Vigor, 1 Outland
Liberator, 4 Thoughtseize.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Choke, 2 Erode | 1 Collector Ouphe, 1 Gaddock Teeg, 1 Sylvan Safekeeper, 1 Icetill Explorer | - |
| burn | 2 Erode, 1 Outland Liberator | 1 Collector Ouphe, 1 Gaddock Teeg, 1 Sylvan Safekeeper | - |
| death_and_taxes | 2 Force of Vigor, 2 Erode, 1 Outland Liberator | 1 Gaddock Teeg, 1 Keen-Eyed Curator, 1 Icetill Explorer, 1 Sylvan Safekeeper, 1 Frenzied Baloth | - |
| lands_top8 | 2 Force of Vigor, 1 Disruptor Flute, 1 Outland Liberator | 3 Orcish Bowmasters, 1 Sylvan Safekeeper | - |
| reanimator | 4 Thoughtseize, 2 Doorkeeper Thrull | 1 Gaddock Teeg, 1 Sylvan Safekeeper, 1 Icetill Explorer, 1 Renegade Rallier, 2 Carnivorous Cultivator | - |
| ur_tempo | 2 Choke, 2 Erode, 1 Force of Vigor | 1 Gaddock Teeg, 1 Icetill Explorer, 1 Sylvan Safekeeper, 1 Keen-Eyed Curator, 1 Renegade Rallier | - |
| uw_control | 2 Choke, 3 Thoughtseize, 1 Force of Vigor | 2 Swords to Plowshares, 1 Prismatic Ending, 1 Icetill Explorer, 1 Renegade Rallier, 1 Carnivorous Cultivator | - |
| wrb_energy | 2 Erode, 1 Force of Vigor | 1 Collector Ouphe, 1 Gaddock Teeg, 1 Sylvan Safekeeper | - |
| wubg_doomsday | 4 Thoughtseize, 3 Doorkeeper Thrull, 2 Choke | 2 Swords to Plowshares, 1 Prismatic Ending, 1 Keen-Eyed Curator, 2 Endurance, 1 Sylvan Safekeeper, 1 Icetill Explorer, 1 Renegade Rallier | - |

Notes: Collector Ouphe stays against lands_top8, reanimator, doomsday, DnT, ur_tempo
and uw_control (Mox Diamond, Lotus Petal, Lion's Eye Diamond, Aether Vial, Cori-Steel Cutter's equip,
The One Ring). Doorkeeper Thrull answers Thassa's Oracle and Archon of Cruelty's and Atraxa's enter
triggers, at the price of the deck's own Bowmasters and Badgermole Cub. Keen-Eyed Curator stays against
reanimator and lands_top8. Thalia stays everywhere. Eight of the guided decks have no row against
maverick yet and keep their maindecks against it; uw_control's is the held-out one.

## reanimator

Sideboard: 2 Consign to Memory, 2 Echoing Truth, 2 Fatal Push, 1 Flusterstorm, 2 Force of Negation,
4 Show and Tell, 2 The Creation of Avacyn.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Force of Negation, 1 Flusterstorm, 2 The Creation of Avacyn | 3 Flow State, 2 Careful Study | Show and Tell |
| burn | 4 Show and Tell, 2 The Creation of Avacyn, 2 Fatal Push | 4 Thoughtseize, 3 Otherworldly Gaze, 1 Flow State | Echoing Truth |
| death_and_taxes | 2 Echoing Truth, 2 Consign to Memory, 2 Fatal Push | 3 Flow State, 1 Raph & Mikey, Troublemakers, 2 Careful Study | Show and Tell |
| lands_top8 | 2 Echoing Truth, 2 Force of Negation, 2 Show and Tell | 3 Flow State, 2 Careful Study, 1 Raph & Mikey, Troublemakers | The Creation of Avacyn |
| ur_tempo | 2 Force of Negation, 1 Flusterstorm, 2 Fatal Push | 3 Flow State, 1 Raph & Mikey, Troublemakers, 1 Dark Ritual | Show and Tell |
| uw_control | 2 Force of Negation, 1 Flusterstorm, 2 Echoing Truth, 1 The Creation of Avacyn | 3 Flow State, 2 Careful Study, 1 Raph & Mikey, Troublemakers | The Creation of Avacyn |
| wrb_energy | 2 Echoing Truth, 2 Fatal Push, 2 Show and Tell, 2 The Creation of Avacyn | 3 Otherworldly Gaze, 3 Careful Study, 2 Thoughtseize | Show and Tell |
| wubg_doomsday | 2 Force of Negation, 1 Flusterstorm, 2 Consign to Memory | 3 Flow State, 1 Raph & Mikey, Troublemakers, 1 Dark Ritual | Echoing Truth |

Notes: against graveyard hate (Leyline of the Void in burn, energy and DnT, Surgical Extraction, Cage,
Endurance) the deck shifts toward Show and Tell and The Creation of Avacyn, which
put the creature in without the graveyard. Containment Priest (uw_control, energy, DnT) stops Show
and Tell and Avacyn too, which is why Echoing Truth comes in there. Echoing Truth also clears all
four Spheres of Resistance or a Deafening Silence at once.

## ur_tempo

Sideboard: 1 Brazen Borrower, 2 Consign to Memory, 1 Echoing Truth, 1 Force of Negation, 1 Grafdigger's
Cage, 1 Hydroblast, 1 Meltdown, 2 Price of Progress, 2 Pyroblast, 1 Rough // Tumble, 1 Surgical
Extraction, 1 Unholy Heat.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Pyroblast, 1 Brazen Borrower, 1 Grafdigger's Cage | 2 Mishra's Bauble, 2 Daze | Force of Negation |
| burn | 1 Hydroblast, 1 Unholy Heat, 1 Rough // Tumble | 2 Daze, 1 Mishra's Bauble | Consign to Memory |
| death_and_taxes | 1 Unholy Heat, 1 Meltdown, 1 Rough // Tumble, 1 Brazen Borrower | 4 Daze | Consign to Memory |
| lands_top8 | 2 Price of Progress, 1 Meltdown, 2 Consign to Memory, 1 Surgical Extraction | 3 Lightning Bolt, 1 Unholy Heat, 2 Daze | Force of Negation |
| reanimator | 1 Surgical Extraction, 1 Grafdigger's Cage, 2 Consign to Memory, 2 Pyroblast | 3 Lightning Bolt, 1 Unholy Heat, 2 Mishra's Bauble | Force of Negation |
| uw_control | 2 Pyroblast, 1 Brazen Borrower, 1 Force of Negation, 1 Consign to Memory | 1 Unholy Heat, 2 Mishra's Bauble, 1 Prismari Charm, 1 Daze | Price of Progress |
| wrb_energy | 1 Hydroblast, 1 Unholy Heat, 1 Rough // Tumble | 2 Daze, 1 Mishra's Bauble | Consign to Memory |
| wubg_doomsday | 1 Force of Negation, 2 Consign to Memory, 2 Pyroblast | 3 Lightning Bolt, 1 Unholy Heat, 1 Prismari Charm | Surgical Extraction |

Notes: the Unholy Heat cut is the main-deck copy; the sideboard copy is a second one. Pyroblast is
for the blue decks, Hydroblast for the red ones (burn, energy). Price of Progress punishes a
nonbasic mana base (lands, uw_control). Rough clears small ground creatures (DnT, burn, energy).
Meltdown hits Vial and the Equipment (DnT), Mox Diamond and Sphere of Resistance (lands). Prismari
Charm stays against reanimator (a reanimated creature) and lands (Marit Lage).

## uw_control

Sideboard: 4 Consign to Memory, 2 Containment Priest, 2 Flusterstorm, 1 Lavinia, Azorius Renegade,
2 Prismatic Ending, 1 Surgical Extraction, 2 Triumph of Saint Katherine, 1 Wrath of the Skies.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Flusterstorm, 1 Lavinia, Azorius Renegade, 1 Surgical Extraction | 2 Wrath of the Skies, 2 Tune the Narrative | Consign to Memory |
| burn | 2 Triumph of Saint Katherine, 2 Prismatic Ending, 2 Containment Priest | 2 Lorien Revealed, 2 Tune the Narrative, 1 Spell Snare, 1 Geist of Saint Traft | Consign to Memory |
| death_and_taxes | 1 Wrath of the Skies, 2 Prismatic Ending | 2 Force of Negation, 1 Spell Pierce | Triumph of Saint Katherine |
| lands_top8 | 2 Prismatic Ending, 2 Consign to Memory, 1 Surgical Extraction, 1 Lavinia, Azorius Renegade | 2 Wrath of the Skies, 1 Spell Snare, 1 Geist of Saint Traft, 2 Tune the Narrative | Consign to Memory |
| maverick | 1 Wrath of the Skies, 2 Prismatic Ending | 2 Force of Negation, 1 Spell Pierce | Containment Priest |
| reanimator | 2 Containment Priest, 1 Surgical Extraction, 2 Consign to Memory, 2 Flusterstorm | 2 Wrath of the Skies, 1 Geist of Saint Traft, 2 Tune the Narrative, 1 Lorien Revealed, 1 Murktide Regent | Lavinia, Azorius Renegade |
| ur_tempo | 2 Flusterstorm, 2 Prismatic Ending, 1 Lavinia, Azorius Renegade | 2 Tune the Narrative, 2 Lorien Revealed, 1 Teferi, Time Raveler | Consign to Memory |
| wrb_energy | 1 Wrath of the Skies, 2 Prismatic Ending, 1 Triumph of Saint Katherine | 2 Force of Negation, 1 Spell Pierce, 1 Lorien Revealed | Triumph of Saint Katherine |
| wubg_doomsday | 2 Flusterstorm, 1 Lavinia, Azorius Renegade, 2 Consign to Memory, 1 Surgical Extraction | 2 Wrath of the Skies, 2 Swords to Plowshares, 1 Geist of Saint Traft, 1 Murktide Regent | Consign to Memory |

Notes: Lavinia counters spells cast without mana (Force of Will, Daze, Lotus Petal, Mox Diamond,
Lion's Eye Diamond).
Swords stays against lands (Marit Lage) and reanimator.

## wrb_energy

Sideboard: 1 Clarion Conqueror, 2 Containment Priest, 1 Damping Sphere, 1 Deafening Silence,
1 Disruptor Flute, 4 Leyline of the Void, 1 Null Rod, 2 Pyroblast, 1 Stony Silence,
1 Wrath of the Skies.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Pyroblast, 1 Clarion Conqueror | 2 Cabal Therapy, 1 Goblin Bombardment | Deafening Silence |
| burn | 1 Clarion Conqueror, 2 Containment Priest, 1 Disruptor Flute | 4 Thoughtseize | Deafening Silence |
| death_and_taxes | 1 Stony Silence, 1 Clarion Conqueror | 1 Cabal Therapy, 1 Thoughtseize | Containment Priest |
| lands_top8 | 4 Leyline of the Void, 1 Null Rod, 1 Disruptor Flute | 3 Orcish Bowmasters, 2 Cabal Therapy, 1 Goblin Bombardment | Clarion Conqueror |
| reanimator | 4 Leyline of the Void, 2 Containment Priest | 2 Goblin Bombardment, 1 Static Prison, 2 Amped Raptor, 1 Cabal Therapy | Disruptor Flute |
| ur_tempo | 2 Pyroblast, 1 Null Rod | 1 Static Prison, 1 Cabal Therapy, 1 Goblin Bombardment | Deafening Silence |
| uw_control | 2 Pyroblast, 1 Clarion Conqueror | 1 Goblin Bombardment, 1 Static Prison, 1 Swords to Plowshares | Null Rod |
| wubg_doomsday | 1 Deafening Silence, 1 Damping Sphere, 1 Disruptor Flute, 2 Pyroblast | 4 Swords to Plowshares, 1 Goblin Bombardment | Clarion Conqueror |

Notes: Clarion Conqueror stops activated abilities of artifacts, creatures and planeswalkers, mana
abilities included: the Stoneforge package and Vial, The One Ring, Teferi, Kaito, Lion's Eye Diamond, Lotus Petal. Hexing Squelcher (spells can't be
countered) stays against the blue decks.

## wubg_doomsday

Sideboard: 3 Carpet of Flowers, 3 Force of Negation, 1 Grafdigger's Cage, 1 Jace, Wielder of
Mysteries, 1 Plains, 3 Prismatic Ending, 3 Swords to Plowshares.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 3 Force of Negation, 2 Carpet of Flowers | 1 Edge of Autumn, 1 Street Wraith, 1 Quantum Riddler, 2 Teferi, Time Raveler | - |
| burn | 3 Swords to Plowshares, 2 Prismatic Ending, 1 Force of Negation | 3 Veil of Summer, 2 Daze, 1 Consider | - |
| death_and_taxes | 3 Prismatic Ending, 2 Swords to Plowshares, 1 Jace, Wielder of Mysteries | 3 Veil of Summer, 2 Daze, 1 Consider | - |
| lands_top8 | 2 Force of Negation, 2 Prismatic Ending, 2 Swords to Plowshares | 3 Veil of Summer, 2 Daze, 1 Edge of Autumn | - |
| reanimator | 3 Force of Negation, 1 Grafdigger's Cage | 1 Edge of Autumn, 1 Street Wraith, 1 Quantum Riddler, 1 Consider | - |
| ur_tempo | 3 Force of Negation, 1 Jace, Wielder of Mysteries | 1 Edge of Autumn, 1 Street Wraith, 1 Quantum Riddler, 1 Consider | - |
| uw_control | 3 Force of Negation, 1 Jace, Wielder of Mysteries, 1 Carpet of Flowers | 1 Edge of Autumn, 1 Street Wraith, 1 Quantum Riddler, 1 Consider, 1 Daze | - |
| wrb_energy | 3 Swords to Plowshares, 3 Prismatic Ending | 2 Daze, 1 Veil of Summer, 1 Edge of Autumn, 1 Consider, 1 Street Wraith | - |

Notes: Jace is a second win condition against Surgical Extraction or Consign to Memory on Thassa's
Oracle. Prismatic Ending and Swords need white: Scrubland, Tundra and the fetches find it; the
sideboard Plains is not used (land counts stay fixed). Veil of Summer goes
out against decks with neither blue nor black (burn, DnT, lands).

## Held-out matchups

For the generalisation test the teacher answers Done in these ten matchups (deck vs opponent),
though their rows stay in the tables above:

| Deck | vs |
|---|---|
| bug | reanimator |
| burn | uw_control |
| death_and_taxes | lands_top8 |
| lands_top8 | wrb_energy |
| maverick | ur_tempo |
| reanimator | wubg_doomsday |
| ur_tempo | bug |
| uw_control | maverick |
| wrb_energy | death_and_taxes |
| wubg_doomsday | burn |

Each deck is held out once on each side, and each held-out matchup has a held-in one calling for
similar cards (burn against uw_control brings in Roiling Vortex as against bug and Smash to
Smithereens as against lands), so a policy that learned what the cards are for can board the
held-out matchups unseen.
