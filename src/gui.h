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

// C-API card info accessors (implemented in gui_card_info.cpp)
#ifdef __cplusplus
extern "C" {
#endif
const char* gui_card_name(int vocab_idx);
const char* gui_card_oracle(int vocab_idx);
const char* gui_card_type_line(int vocab_idx);
const char* gui_step_name(int step);
int gui_card_base_power(int vocab_idx);
int gui_card_base_toughness(int vocab_idx);
#ifdef __cplusplus
}
#endif

#endif /* GUI_H */
