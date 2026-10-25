#ifndef PARSE_REPORT_H
#define PARSE_REPORT_H

#include <string>
#include <vector>

// ── --parse-report: what the parser does not model in a card's script (plays no game) ──────────
//
// Each `--parse-report <arg>` (repeatable) names cards: a card name as a deck line spells it
// (load_card's resolution: a double-faced card by either face's name), "@vocab" (every card of
// src/card_vocab.h) or "@FILE" (FILE's cards: a .dk deck's main deck and sideboard, else one card
// name per line; relative to the working directory). Each distinct script is parsed afresh with
// the parser's findings collected (parse.h, ParseFinding), and with it every token script it names
// (TokenScript$, and the stems the engine synthesizes for Amass, Investigate and Mobilize), and one
// JSON object per card is printed on stdout, in argument order:
//
//   {"query": <arg's name>, "card": <front face name>, "script": <path under resources/>,
//    "faces": [<face names>], "layout": "normal"|"transform"|"modal"|"split", "vocab": [<ids>],
//    "tokens": [{"script": <stem>, "present": <bool>}], "clean": <bool>,
//    "findings": [{"kind": ..., "name": ..., "detail": ..., "line": ...}, ...]}
//
// A token script's findings carry "token:<stem>" before their line; a token script that is not on
// disk is a finding of kind "token_script". A name with no script prints {"query": ..., "card":
// null, "error": "no script", "clean": false}. The finding kinds:
//
//   category        an ability category with no effect handler (it resolves as a no-op)
//   ability_param   an ability param no parse branch reads ("name" the key)
//   amount_svar     an amount names an SVar the script lacks
//   svar_missing    SubAbility$ / Execute$ / Choices$ / … names an SVar the script lacks
//   trigger_mode    a trigger Mode$ with no event binding (the trigger is dropped)
//   trigger_form    a bound mode whose form here binds no event (dropped)
//   trigger_param   a trigger param no branch reads
//   trigger_phase   a Phase$ step with no event
//   trigger_zone    TriggerZones$ / Origin$ / Destination$ zones the binding does not filter on
//   static_mode     an S: mode the engine applies nowhere
//   static_param    a static param no branch reads
//   static_form     a form of a modelled static mode that is dropped
//   replacement     an R: line (or an Effect's ReplacementEffects$ SVar) that yields no replacement
//   keyword         a plain keyword no rule reads
//   cost            a cost token with no meaning
//   layout          an AlternateMode the engine has no layout for
//   type_token      a type-line token that is no card type, subtype or supertype the engine knows
//                   (type_constants.h; a game reads it as a subtype with a non-fatal error)
//   token_script    a token script the card names that is not on disk
//
// A clean report does not prove the card plays correctly (a modelled param can still be read only
// in the forms its first cards needed); a finding names something the engine does not do.
// A summary "parse-report: <n> card(s), <k> with findings, <m> without a script" goes to stderr.
// Exit status 0 when every card is clean, 1 otherwise, 2 for an unreadable @FILE.

extern std::vector<std::string> parse_report_args;  // --parse-report values, in order

// True when --parse-report was given.
bool parse_report_requested();

// Runs the report; the process exit status.
int run_parse_report();

#endif /* PARSE_REPORT_H */
