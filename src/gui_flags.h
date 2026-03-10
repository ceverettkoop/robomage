#ifndef GUI_FLAGS_H
#define GUI_FLAGS_H

// Out-of-band flag values returned by get_input() / gui_cmd.
// Negative integers that do not correspond to any legal action index.
// Shared between input_logger.h (C++) and gui.c (C99).
#define FLAG_QUIT         -3
#define FLAG_RESTART_GAME -4

#endif /* GUI_FLAGS_H */
