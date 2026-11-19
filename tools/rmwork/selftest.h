#ifndef RMWORK_SELFTEST_H
#define RMWORK_SELFTEST_H

// ── rmwork --selftest: the shell's checks, headless ─────────────────────────────────────────────
//
// An ImGui context with no window or renderer runs App::frame(): every preset in both modes (the
// preset's tools open and no other, each docked into the mode's dockspace in its tab group, in
// order, the first tab selected, the groups at the preset's split ratios), each mode keeping its
// own layout across switches and an ini round trip, a closed tool reopening where it was, a tool
// outside the preset joining the first group, every embedded font family × weight becoming the
// frame's font at the size asked for, the bars and presets at the smallest and largest font sizes,
// every colour theme drawing both modes with its palette's colours and meeting the contrast
// minimums, and the settings file (the font and theme and their fallbacks too) round-tripped in a
// temporary $XDG_CONFIG_HOME, and the start page (selftest_start.h). Prints "RMWORK-SELFTEST: N
// checks, M failures"; nonzero exit on a failure.

namespace rmwork {

int run_selftest();

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_H */
