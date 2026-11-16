#ifndef RMWORK_MODAL_H
#define RMWORK_MODAL_H

#include "imgui.h"

// ── What the workbench's modal dialogs share: their frame, footer, notes and field rows ─────────

namespace rmwork {

// The current font size over the default's, for sizes laid out at the default.
float em_scale();
// Centres the next modal, `size` (at the default font size) scaled to the font and kept inside the
// viewport, and gives it the theme's dialog title bar (pop with ImGui::PopStyleColor(3) right after
// the Begin, which draws the title).
void begin_modal_frame(ImVec2 size);
// Esc closes the current modal.
bool close_requested();
// The height a dialog's footer (a separator and a row of buttons) takes.
float footer_height();
// A button ending at the right edge of the content region, from the cursor's line.
bool right_aligned_button(const char *label);
// A wrapped line of secondary text.
void note(const char *text);
// A line of text in `color`, wrapped.
void colored_note(const char *text, ImU32 color);
// A field's label on the frame's text line, the field starting `field_x` from the line's start.
void field_label(const char *label, float field_x);
// The width of a character of the current font ("M").
float char_width();

}  // namespace rmwork

#endif /* RMWORK_MODAL_H */
