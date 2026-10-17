#ifndef STATE_FORMAT_SELFCHECK_KNOWLEDGE_H
#define STATE_FORMAT_SELFCHECK_KNOWLEDGE_H

#include <string>

#include "../components/zone.h"

// ── --selfcheck-knowledge: the knowledge sidecar against the truth (docs/knowledge_store.md §7) ─
//
// Implies --selfcheck-state (state_format/selfcheck.h), whose own knowledge instance per seat
// (ObsConsumer::SELFCHECK) appends k_* tables to every seat frame it builds. On top of its checks:
//
//   soundness  at every decision, each seat's k_* tables, read back from the frame's bytes, against
//              the true world built from the engine state itself (not from a frame): every card
//              object's true identity, hand, sideboard and library order, and each face-down object
//              by the seat's handle. Every reported fact must hold (knowledge/truth.h: pool sizes,
//              k_pool_card ranges, block pools included, k_position cards and candidates,
//              k_known_hand, k_seen); zero tolerance;
//   snapshot   when the self-check's snapshot check restores a saved decision, every knowledge
//              instance's serialised state (Knowledge::serialize, every writer and seat) equals the
//              one at the save; the frames re-fed there are compared, k_* tables included, by the
//              self-check's own snapshot check.
//
// A failure is an `ERROR: selfcheck-knowledge <check>: viewer …, decision …: …` line; a
// `SELFCHECK-KNOWLEDGE: decisions=… facts=… errors=…` summary goes to stderr at match end.

// Whether --selfcheck-knowledge is in effect (with the state self-check on).
bool selfcheck_knowledge_enabled();

// The soundness check of one decision: `frame_a`, `frame_b` are the seats' STATE frames as the
// self-check built them, k_* tables included.
void selfcheck_knowledge_decision(const std::string &frame_a, const std::string &frame_b);

// The snapshot check's two ends: the instances' state as saved, and as restored.
void selfcheck_knowledge_saved();
void selfcheck_knowledge_restored();

// The match is over: the summary line.
void selfcheck_knowledge_match_over();

#endif /* STATE_FORMAT_SELFCHECK_KNOWLEDGE_H */
