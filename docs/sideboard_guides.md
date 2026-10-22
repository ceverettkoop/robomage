# League sideboard guides

Sideboarding for the `scripted:hard+sb` teacher tier (`docs/rnad_training_plan.md` §6.2):
for each league deck, what it brings in and cuts against each other league deck, and what it
changes on the draw. These are a prior for the warm start, not the target: the trained policy is
meant to learn sideboarding that carries beyond this table.

**Status:** reviewed 2026-10-21; for training. The machine-readable copy the tier reads is
`python/sideboard_guides.json` (draw lands, every table row and the held-out matchups, card names
as the league deck files spell them); `python/pycheck.py` (the gate's `smoke` tier) checks that it
agrees with the tables here and with the deck files.

## How the guides are applied

- A guide is a **target configuration** per (own deck, opponent deck, on the play or on the draw).
  At every sideboarding phase the teacher swaps from its current 75 to that target: game 3 starts
  from game 2's configuration, and play/draw can flip between games 2 and 3, so the land rule is
  undone or applied as needed.
- The opponent's deck is known from its registered list (open decklists). An opponent list that is
  not a league deck, a matchup a guide leaves out, or a held-out matchup (below) gets Done (no
  swaps).
- **On the play:** the matchup's *In* and *Out*.
- **On the draw:** the same, plus one more swap: the deck's draw land out and the matchup's *Draw +1*
  card in. gw_maverick, lands and wubg_doomsday cut no land on the draw: their guides are the same
  on the play and on the draw.
- Every swap is one card in and one out (the engine's balance rule), so *In* and *Out* have equal
  counts. All of a deck's phases stay under the engine's 15-swap cap.

## Draw land per deck

| Deck | Cut on the draw |
|---|---|
| bug | 1 Swamp |
| burn | 1 Mountain |
| bw_dnt | 1 Plains |
| gw_maverick | none |
| lands | none |
| reanimator | 1 Swamp |
| ur_delver | 1 Thundering Falls |
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
| bw_dnt | 2 Pernicious Deed, 1 Toxic Deluge | 2 Thoughtseize, 1 Lorien Revealed | Force of Vigor |
| gw_maverick | 2 Pernicious Deed, 1 Toxic Deluge | 2 Thoughtseize, 1 Lorien Revealed | Force of Vigor |
| lands | 1 Force of Vigor, 1 Surgical Extraction, 2 Consign to Memory, 1 Pernicious Deed | 2 Orcish Bowmasters, 2 Fatal Push, 1 Witherbloom Command | Null Rod |
| reanimator | 1 Surgical Extraction, 2 Consign to Memory, 1 Force of Negation, 2 Endurance | 3 Fatal Push, 2 Witherbloom Command, 1 Lorien Revealed | Veil of Summer |
| ur_delver | 1 Hydroblast, 1 Null Rod, 1 Toxic Deluge | 2 Thoughtseize, 1 Life from the Loam | Veil of Summer |
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
| bw_dnt | 3 Searing Blood, 2 Dead // Gone, 2 Smash to Smithereens | 4 Eidolon of the Great Revel, 2 Lava Spike, 1 Light Up the Stage | Smash to Smithereens |
| gw_maverick | 3 Searing Blood, 2 Dead // Gone | 4 Eidolon of the Great Revel, 1 Light Up the Stage | Roiling Vortex |
| lands | 4 Leyline of the Void, 2 Smash to Smithereens | 2 Exquisite Firecraft, 1 Light Up the Stage, 3 Skewer the Critics | Smash to Smithereens |
| reanimator | 4 Leyline of the Void, 2 Roiling Vortex | 2 Exquisite Firecraft, 1 Light Up the Stage, 3 Skewer the Critics | Dead // Gone |
| ur_delver | 3 Searing Blood, 2 Roiling Vortex | 1 Light Up the Stage, 3 Skewer the Critics, 1 Rift Bolt | Smash to Smithereens |
| uw_control | 2 Roiling Vortex, 2 Smash to Smithereens | 1 Light Up the Stage, 3 Skewer the Critics | Searing Blood |
| wrb_energy | 3 Searing Blood, 1 Dead // Gone, 2 Roiling Vortex | 4 Eidolon of the Great Revel, 1 Lava Spike, 1 Light Up the Stage | Dead // Gone |
| wubg_doomsday | 2 Roiling Vortex | 1 Light Up the Stage, 1 Skewer the Critics | Dead // Gone |

Notes: Roiling Vortex punishes free spells (Force of Will, Daze, Lotus Petal, Lion's Eye Diamond)
and stops lifegain (Uro, Guide of Souls, Ocelot Pride). Smash hits The One Ring, Cori-Steel Cutter,
the Stoneforge package and Vial. Gone answers a reanimated Atraxa or Griselbrand.

## bw_dnt

Sideboard: 1 Containment Priest, 3 Deafening Silence, 2 Disruptor Flute, 2 Erode, 1 Faerie Macabre,
1 Grafdigger's Cage, 1 Surgical Extraction, 3 Wrath of the Skies, 1 Yorion, Sky Nomad (the
companion: never boarded in).

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Erode | 1 Meteor Sword, 1 Cloak and Dagger, Entwined | Deafening Silence |
| burn | 2 Erode, 2 Disruptor Flute | 4 Thoughtseize | Containment Priest |
| gw_maverick | 2 Erode, 1 Grafdigger's Cage | 2 Thoughtseize, 1 Orcish Bowmasters | Disruptor Flute |
| lands | 1 Surgical Extraction, 1 Faerie Macabre, 2 Disruptor Flute | 3 Orcish Bowmasters, 1 Meteor Sword | Deafening Silence |
| reanimator | 1 Containment Priest, 1 Faerie Macabre, 1 Grafdigger's Cage, 1 Surgical Extraction, 2 Disruptor Flute | 2 Pre-War Formalwear, 1 Meteor Sword, 1 Cloak and Dagger, Entwined, 1 Flickerwisp, 1 Orcish Bowmasters | Deafening Silence |
| ur_delver | 2 Erode, 1 Deafening Silence | 1 Meteor Sword, 1 Cloak and Dagger, Entwined, 1 Thoughtseize | Deafening Silence |
| uw_control | 2 Deafening Silence, 2 Erode | 1 Meteor Sword, 1 Cloak and Dagger, Entwined, 2 Swords to Plowshares | Deafening Silence |
| wrb_energy | 2 Erode | 2 Thoughtseize | Disruptor Flute |
| wubg_doomsday | 3 Deafening Silence, 2 Disruptor Flute, 1 Surgical Extraction | 3 Swords to Plowshares, 1 Meteor Sword, 1 Cloak and Dagger, Entwined, 1 Lion Sash | Erode |

Notes: Containment Priest also exiles the deck's own blinked and Formalwear-returned creatures, so
it comes in with the Formalwear and Flickerwisp cuts. Flute names Reanimate (reanimator), Doomsday,
Thespian's Stage (lands), Lightning Bolt (burn), Goblin Bombardment (energy), Knight of the
Reliquary (maverick); which name to pick is the teacher's choice at the name prompt. Karakas stays
in every matchup (Atraxa, Griselbrand, Marit Lage, Uro).

## gw_maverick

Sideboard: 2 Choke, 3 Deafening Silence, 2 Endurance, 2 Erode, 3 Force of Vigor, 1 Gaddock Teeg,
1 Mole Man, Moloid Master, 1 Sheltered by Ghosts.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Choke, 2 Erode, 1 Deafening Silence | 1 Collector Ouphe, 1 Gaddock Teeg, 2 Once Upon a Time, 1 Icetill Explorer | - |
| burn | 1 Sheltered by Ghosts, 2 Endurance | 1 Collector Ouphe, 1 Gaddock Teeg, 1 Once Upon a Time | - |
| bw_dnt | 3 Force of Vigor, 2 Erode | 1 Gaddock Teeg, 1 Keen-Eyed Curator, 3 Once Upon a Time | - |
| lands | 3 Force of Vigor, 2 Endurance | 2 Swords to Plowshares, 1 Scythecat Cub, 1 Icetill Explorer, 1 Outland Liberator | - |
| reanimator | 2 Endurance, 2 Deafening Silence | 1 Icetill Explorer, 1 Outland Liberator, 2 Once Upon a Time | - |
| ur_delver | 2 Choke, 2 Erode, 1 Force of Vigor | 1 Gaddock Teeg, 1 Keen-Eyed Curator, 1 Icetill Explorer, 2 Once Upon a Time | - |
| uw_control | 2 Choke, 1 Force of Vigor, 1 Gaddock Teeg, 2 Erode | 1 Collector Ouphe, 1 Keen-Eyed Curator, 2 Once Upon a Time, 1 Outland Liberator, 1 Scythecat Cub | - |
| wrb_energy | 2 Erode, 1 Sheltered by Ghosts, 1 Force of Vigor | 1 Collector Ouphe, 1 Gaddock Teeg, 2 Once Upon a Time | - |
| wubg_doomsday | 3 Deafening Silence, 2 Choke | 2 Swords to Plowshares, 1 Keen-Eyed Curator, 2 Endurance | - |

Notes: Collector Ouphe stays against lands, reanimator and doomsday (Mox Diamond, Lotus Petal,
Lion's Eye Diamond). Keen-Eyed Curator stays against reanimator and lands. Against uw_control the
sideboard Teeg joins the main one (The One Ring, Wrath of the Skies' X). Thalia stays everywhere.

## lands

Sideboard: 1 Damping Sphere, 3 Endurance, 3 Faerie Macabre, 3 Force of Vigor, 1 Pithing Needle,
2 Swords to Plowshares, 2 Twinshot Sniper.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 1 Pithing Needle, 2 Endurance | 1 Skateboard, 1 Expedition Map, 1 Malevolent Rumble | - |
| burn | 3 Endurance, 2 Twinshot Sniper, 2 Swords to Plowshares | 1 Pithing Needle, 1 Grafdigger's Cage, 1 Expedition Map, 1 Skateboard, 3 Crop Rotation | - |
| bw_dnt | 3 Force of Vigor, 1 Pithing Needle, 2 Swords to Plowshares | 2 Sphere of Resistance, 1 Grafdigger's Cage, 1 Skateboard, 1 Expedition Map, 1 Malevolent Rumble | - |
| gw_maverick | 2 Swords to Plowshares, 2 Twinshot Sniper, 1 Pithing Needle | 1 Skateboard, 1 Expedition Map, 2 Sphere of Resistance, 1 Malevolent Rumble | - |
| reanimator | 3 Faerie Macabre, 2 Swords to Plowshares | 1 Skateboard, 1 Expedition Map, 1 Pithing Needle, 2 Malevolent Rumble | - |
| ur_delver | 1 Pithing Needle, 2 Swords to Plowshares, 2 Twinshot Sniper | 1 Skateboard, 1 Expedition Map, 1 Grafdigger's Cage, 2 Malevolent Rumble | - |
| uw_control | 1 Pithing Needle, 3 Force of Vigor, 1 Damping Sphere | 1 Grafdigger's Cage, 1 Skateboard, 2 Malevolent Rumble, 1 Expedition Map | - |
| wrb_energy | 2 Swords to Plowshares, 2 Twinshot Sniper, 1 Force of Vigor | 1 Grafdigger's Cage, 1 Skateboard, 1 Expedition Map, 2 Sphere of Resistance | - |
| wubg_doomsday | 1 Damping Sphere, 1 Pithing Needle, 1 Force of Vigor | 2 Swords to Plowshares, 1 Grafdigger's Cage | - |

Notes: the sideboard Pithing Needle names Wasteland or Karakas (DnT, maverick), The One Ring or
Teferi (uw_control), Kaito (bug), Lion's Eye Diamond (doomsday). Grafdigger's Cage stays against
bug (Uro's escape), maverick (Green Sun's Zenith) and reanimator.

## reanimator

Sideboard: 2 Consign to Memory, 2 Echoing Truth, 2 Fatal Push, 1 Flusterstorm, 2 Force of Negation,
4 Show and Tell, 2 The Creation of Avacyn.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Force of Negation, 1 Flusterstorm, 2 The Creation of Avacyn | 3 Flow State, 2 Careful Study | Show and Tell |
| burn | 4 Show and Tell, 2 The Creation of Avacyn, 2 Fatal Push | 4 Thoughtseize, 3 Otherworldly Gaze, 1 Flow State | Echoing Truth |
| bw_dnt | 2 Echoing Truth, 2 Consign to Memory, 2 Fatal Push | 3 Flow State, 1 Raph & Mikey, Troublemakers, 2 Careful Study | Show and Tell |
| gw_maverick | 2 Fatal Push, 2 Echoing Truth, 2 Show and Tell | 3 Flow State, 2 Careful Study, 1 Raph & Mikey, Troublemakers | The Creation of Avacyn |
| lands | 2 Echoing Truth, 2 Force of Negation, 2 Show and Tell | 3 Flow State, 2 Careful Study, 1 Raph & Mikey, Troublemakers | The Creation of Avacyn |
| ur_delver | 2 Force of Negation, 1 Flusterstorm, 2 Fatal Push | 3 Flow State, 1 Raph & Mikey, Troublemakers, 1 Dark Ritual | Show and Tell |
| uw_control | 2 Force of Negation, 1 Flusterstorm, 2 Echoing Truth, 1 The Creation of Avacyn | 3 Flow State, 2 Careful Study, 1 Raph & Mikey, Troublemakers | The Creation of Avacyn |
| wrb_energy | 2 Echoing Truth, 2 Fatal Push, 2 Show and Tell, 2 The Creation of Avacyn | 3 Otherworldly Gaze, 3 Careful Study, 2 Thoughtseize | Show and Tell |
| wubg_doomsday | 2 Force of Negation, 1 Flusterstorm, 2 Consign to Memory | 3 Flow State, 1 Raph & Mikey, Troublemakers, 1 Dark Ritual | Echoing Truth |

Notes: against graveyard hate (Leyline of the Void in burn and energy, Surgical Extraction, Cage,
Endurance, Faerie Macabre) the deck shifts toward Show and Tell and The Creation of Avacyn, which
put the creature in without the graveyard. Containment Priest (uw_control, energy, DnT) stops Show
and Tell and Avacyn too, which is why Echoing Truth comes in there. Echoing Truth also clears all
four Spheres of Resistance or a Deafening Silence at once.

## ur_delver

Sideboard: 2 Blue Elemental Blast, 2 Consign to Memory, 2 Force of Negation, 1 Grafdigger's Cage,
1 Meltdown, 2 Price of Freedom, 3 Red Elemental Blast, 1 Surgical Extraction, 1 Unholy Heat.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 3 Red Elemental Blast, 1 Grafdigger's Cage | 2 Mishra's Bauble, 2 Daze | Force of Negation |
| burn | 2 Blue Elemental Blast, 1 Unholy Heat | 2 Daze, 1 Mishra's Bauble | Consign to Memory |
| bw_dnt | 1 Unholy Heat, 1 Meltdown, 2 Price of Freedom | 4 Daze | Consign to Memory |
| gw_maverick | 1 Unholy Heat, 1 Grafdigger's Cage, 1 Price of Freedom | 2 Daze, 1 Mishra's Bauble | Price of Freedom |
| lands | 2 Price of Freedom, 1 Meltdown, 2 Consign to Memory, 1 Surgical Extraction | 3 Lightning Bolt, 1 Unholy Heat, 2 Daze | Force of Negation |
| reanimator | 1 Surgical Extraction, 1 Grafdigger's Cage, 2 Consign to Memory, 2 Red Elemental Blast | 3 Lightning Bolt, 1 Unholy Heat, 2 Mishra's Bauble | Force of Negation |
| uw_control | 3 Red Elemental Blast, 2 Force of Negation | 1 Unholy Heat, 2 Mishra's Bauble, 1 Into the Flood Maw, 1 Daze | Price of Freedom |
| wrb_energy | 2 Blue Elemental Blast, 1 Unholy Heat | 2 Daze, 1 Mishra's Bauble | Consign to Memory |
| wubg_doomsday | 2 Force of Negation, 2 Consign to Memory, 1 Red Elemental Blast | 3 Lightning Bolt, 1 Unholy Heat, 1 Into the Flood Maw | Surgical Extraction |

Notes: the Unholy Heat cut is the main-deck copy; the sideboard copy is a second one. Price of
Freedom hits Dark Depths, Thespian's Stage, Urza's Saga and Mox Diamond (lands), Karakas and Vial
(DnT), Gaea's Cradle (maverick), The One Ring (uw_control). Into the Flood Maw stays against
reanimator (a reanimated creature) and lands (Marit Lage).

## uw_control

Sideboard: 4 Consign to Memory, 2 Containment Priest, 2 Flusterstorm, 1 Lavinia, Azorius Renegade,
2 Prismatic Ending, 1 Surgical Extraction, 2 Triumph of Saint Katherine, 1 Wrath of the Skies.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Flusterstorm, 1 Lavinia, Azorius Renegade, 1 Surgical Extraction | 2 Wrath of the Skies, 2 Tune the Narrative | Consign to Memory |
| burn | 2 Triumph of Saint Katherine, 2 Prismatic Ending, 2 Containment Priest | 2 Lorien Revealed, 2 Tune the Narrative, 1 Spell Snare, 1 Geist of Saint Traft | Consign to Memory |
| bw_dnt | 1 Wrath of the Skies, 2 Prismatic Ending | 2 Force of Negation, 1 Spell Pierce | Triumph of Saint Katherine |
| gw_maverick | 1 Wrath of the Skies, 2 Prismatic Ending | 2 Force of Negation, 1 Spell Pierce | Containment Priest |
| lands | 2 Prismatic Ending, 2 Consign to Memory, 1 Surgical Extraction, 1 Lavinia, Azorius Renegade | 2 Wrath of the Skies, 1 Spell Snare, 1 Geist of Saint Traft, 2 Tune the Narrative | Consign to Memory |
| reanimator | 2 Containment Priest, 1 Surgical Extraction, 2 Consign to Memory, 2 Flusterstorm | 2 Wrath of the Skies, 1 Geist of Saint Traft, 2 Tune the Narrative, 1 Lorien Revealed, 1 Murktide Regent | Lavinia, Azorius Renegade |
| ur_delver | 2 Flusterstorm, 2 Prismatic Ending, 1 Lavinia, Azorius Renegade | 2 Tune the Narrative, 2 Lorien Revealed, 1 Teferi, Time Raveler | Consign to Memory |
| wrb_energy | 1 Wrath of the Skies, 2 Prismatic Ending, 1 Triumph of Saint Katherine | 2 Force of Negation, 1 Spell Pierce, 1 Lorien Revealed | Triumph of Saint Katherine |
| wubg_doomsday | 2 Flusterstorm, 1 Lavinia, Azorius Renegade, 2 Consign to Memory, 1 Surgical Extraction | 2 Wrath of the Skies, 2 Swords to Plowshares, 1 Geist of Saint Traft, 1 Murktide Regent | Consign to Memory |

Notes: Lavinia counters spells cast without mana (Force of Will, Daze, Lotus Petal, Mox Diamond,
Lion's Eye Diamond). Containment Priest against maverick exiles creatures Green Sun's Zenith puts
in. Swords stays against lands (Marit Lage) and reanimator.

## wrb_energy

Sideboard: 1 Clarion Conqueror, 2 Containment Priest, 1 Damping Sphere, 1 Deafening Silence,
1 Disruptor Flute, 4 Leyline of the Void, 1 Null Rod, 2 Pyroblast, 1 Stony Silence,
1 Wrath of the Skies.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 2 Pyroblast, 1 Clarion Conqueror | 2 Cabal Therapy, 1 Goblin Bombardment | Deafening Silence |
| burn | 1 Clarion Conqueror, 2 Containment Priest, 1 Disruptor Flute | 4 Thoughtseize | Deafening Silence |
| bw_dnt | 1 Stony Silence, 1 Clarion Conqueror | 1 Cabal Therapy, 1 Thoughtseize | Containment Priest |
| gw_maverick | 1 Clarion Conqueror, 1 Containment Priest | 2 Cabal Therapy | Containment Priest |
| lands | 4 Leyline of the Void, 1 Null Rod, 1 Disruptor Flute | 3 Orcish Bowmasters, 2 Cabal Therapy, 1 Goblin Bombardment | Clarion Conqueror |
| reanimator | 4 Leyline of the Void, 2 Containment Priest | 2 Goblin Bombardment, 1 Static Prison, 2 Amped Raptor, 1 Cabal Therapy | Disruptor Flute |
| ur_delver | 2 Pyroblast, 1 Null Rod | 1 Static Prison, 1 Cabal Therapy, 1 Goblin Bombardment | Deafening Silence |
| uw_control | 2 Pyroblast, 1 Clarion Conqueror | 1 Goblin Bombardment, 1 Static Prison, 1 Swords to Plowshares | Null Rod |
| wubg_doomsday | 1 Deafening Silence, 1 Damping Sphere, 1 Disruptor Flute, 2 Pyroblast | 4 Swords to Plowshares, 1 Goblin Bombardment | Clarion Conqueror |

Notes: Clarion Conqueror stops activated abilities of artifacts, creatures and planeswalkers, mana
abilities included: the Stoneforge package and Vial, Knight of the Reliquary and the mana elves,
The One Ring, Teferi, Kaito, Lion's Eye Diamond, Lotus Petal. Hexing Squelcher (spells can't be
countered) stays against the blue decks.

## wubg_doomsday

Sideboard: 3 Carpet of Flowers, 3 Force of Negation, 1 Grafdigger's Cage, 1 Jace, Wielder of
Mysteries, 1 Plains, 3 Prismatic Ending, 3 Swords to Plowshares.

| vs | In | Out | Draw +1 |
|---|---|---|---|
| bug | 3 Force of Negation, 2 Carpet of Flowers | 1 Edge of Autumn, 1 Street Wraith, 1 Quantum Riddler, 2 Teferi, Time Raveler | - |
| burn | 3 Swords to Plowshares, 2 Prismatic Ending, 1 Force of Negation | 3 Veil of Summer, 2 Daze, 1 Consider | - |
| bw_dnt | 3 Prismatic Ending, 2 Swords to Plowshares, 1 Jace, Wielder of Mysteries | 2 Veil of Summer, 2 Daze, 1 Consider, 1 Edge of Autumn | - |
| gw_maverick | 3 Swords to Plowshares, 3 Prismatic Ending | 3 Veil of Summer, 2 Daze, 1 Consider | - |
| lands | 2 Force of Negation, 2 Prismatic Ending, 2 Swords to Plowshares | 3 Veil of Summer, 2 Daze, 1 Edge of Autumn | - |
| reanimator | 3 Force of Negation, 1 Grafdigger's Cage | 1 Edge of Autumn, 1 Street Wraith, 1 Quantum Riddler, 1 Consider | - |
| ur_delver | 3 Force of Negation, 1 Jace, Wielder of Mysteries | 1 Edge of Autumn, 1 Street Wraith, 1 Quantum Riddler, 1 Consider | - |
| uw_control | 3 Force of Negation, 1 Jace, Wielder of Mysteries, 1 Carpet of Flowers | 1 Edge of Autumn, 1 Street Wraith, 1 Quantum Riddler, 1 Consider, 1 Daze | - |
| wrb_energy | 3 Swords to Plowshares, 3 Prismatic Ending | 2 Daze, 1 Veil of Summer, 1 Edge of Autumn, 1 Consider, 1 Street Wraith | - |

Notes: Jace is a second win condition against Surgical Extraction or Consign to Memory on Thassa's
Oracle. Prismatic Ending and Swords need white: Scrubland, Tundra and the fetches find it; the
sideboard Plains is not used (land counts stay fixed). Veil of Summer goes
out against decks with neither blue nor black (burn, maverick, lands).

## Held-out matchups

For the generalisation test (`docs/rnad_training_plan.md` §6.2) the teacher answers Done in these
ten matchups (deck vs opponent), though their rows stay in the tables above:

| Deck | vs |
|---|---|
| bug | reanimator |
| burn | uw_control |
| bw_dnt | lands |
| gw_maverick | ur_delver |
| lands | wrb_energy |
| reanimator | wubg_doomsday |
| ur_delver | bug |
| uw_control | gw_maverick |
| wrb_energy | bw_dnt |
| wubg_doomsday | burn |

Each deck is held out once as the sideboarding deck and once as the opponent, and each held-out
matchup has a held-in one calling for similar cards (bug against reanimator brings in the
graveyard hate it brings in against lands; burn against uw_control brings in Roiling Vortex as
against bug and Smash to Smithereens as against lands; …), so a policy that learned what the cards are for can board the
held-out matchups without having been shown them.
