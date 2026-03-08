#ifndef GUI_H
#define GUI_H

#include <stdbool.h>
#include "pthread.h"

#define GUI_INPUT_MAX 64

void init_gui();
void gui_set_resource_dir(const char *path);

// C-API log buffer accessors (implemented in cli_output.cpp)
int gui_log_line_count(void);
const char* gui_log_get_line(int idx);
void gui_query_clear(void);
int gui_query_line_count(void);
const char* gui_query_get_line(int idx);

#endif /* GUI_H */
