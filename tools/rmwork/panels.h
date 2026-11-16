#ifndef RMWORK_PANELS_H
#define RMWORK_PANELS_H

#include "layout.h"

// ── The tools' contents ─────────────────────────────────────────────────────────────────────────
//
// No game is loaded yet: each tool shows what it will show; the Value Graph draws its empty axes.

namespace rmwork {

// Draws tool `t`'s body inside its window (between Begin and End).
void draw_tool_body(Tool t);

}  // namespace rmwork

#endif /* RMWORK_PANELS_H */
