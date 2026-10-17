#include "recode.h"

#include <cstdio>

#include "decode.h"
#include "json.h"
#include "schema.h"

static void report(size_t record, const std::string &what);

static void report(size_t record, const std::string &what) {
    std::fprintf(stderr, "ERROR: rmstate read: record %zu: %s\n", record, what.c_str());
}

int rmstate_read_stream(const std::string &path, bool json) {
    FILE *f = path == "-" ? stdin : std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "ERROR: rmstate read: cannot open %s\n", path.c_str());
        return 1;
    }
    rmstate::RecordReader reader(f);
    rmstate::Schema schema;
    bool have_schema = false;
    size_t identical = 0, errors = 0;
    std::string payload;
    rmstate::DecodeResult res;
    while (reader.next(&payload, &res)) {
        const size_t record = reader.count();
        if (payload.size() > 6 &&
            static_cast<uint8_t>(payload[6]) == static_cast<uint8_t>(rmstate::FrameKind::SCHEMA)) {
            rmstate::Schema next;
            res = rmstate::decode_schema_frame(payload, &next);
            if (!res.ok()) {
                report(record, "SCHEMA frame: " + res.message);
                ++errors;
                continue;
            }
            schema = std::move(next);
            have_schema = true;
        }
        if (!have_schema) {
            report(record, "frame before any SCHEMA frame");
            ++errors;
            continue;
        }
        rmstate::DecodedFrame frame;
        res = rmstate::decode_frame(payload, schema, &frame);
        if (!res.ok()) {
            report(record, "decode: " + res.message);
            ++errors;
            continue;
        }
        if (json) {
            const std::string line = rmstate::frame_json(frame, schema) + "\n";
            std::fwrite(line.data(), 1, line.size(), stdout);
        }
        std::string again;
        res = rmstate::encode_frame(frame, &again);
        if (!res.ok()) {
            report(record, "re-encode: " + res.message);
            ++errors;
            continue;
        }
        if (again != payload) {
            size_t at = 0;
            while (at < again.size() && at < payload.size() && again[at] == payload[at]) ++at;
            report(record, "re-encoded bytes differ (sizes " + std::to_string(payload.size()) + " / " +
                               std::to_string(again.size()) + ", first difference at byte " + std::to_string(at) + ")");
            ++errors;
            continue;
        }
        ++identical;
    }
    if (!res.ok()) {
        report(reader.count() + 1, res.message);
        ++errors;
    }
    if (f != stdin) std::fclose(f);
    std::fflush(stdout);
    std::fprintf(stderr, "RMSTATE_READ: records=%zu identical=%zu errors=%zu\n", reader.count(), identical, errors);
    return errors == 0 ? 0 : 1;
}
