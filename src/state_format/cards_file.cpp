#include "cards_file.h"

#include <cstdio>
#include <fstream>

#include "../components/zone.h"
#include "../game_driver.h"
#include "encode.h"
#include "schema.h"

std::string cards_frame_path;

int write_cards_frame_file(const std::string &path) {
    const bool was_closed = closed_decklist;
    closed_decklist = true;
    const std::string records = rmstate::frame_record(encode_schema_frame(Zone::UNKNOWN)) +
                                rmstate::frame_record(encode_cards_frame(Zone::UNKNOWN)) +
                                rmstate::frame_record(encode_metagame_frame(Zone::UNKNOWN));
    closed_decklist = was_closed;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(records.data(), static_cast<std::streamsize>(records.size()));
    if (!f.good()) {
        std::fprintf(stderr, "CARDS-FRAME: cannot write %s\n", path.c_str());
        return 2;
    }
    std::printf("CARDS-FRAME: wrote %s (SCHEMA, CARDS, METAGAME; %zu bytes)\n", path.c_str(), records.size());
    return 0;
}
