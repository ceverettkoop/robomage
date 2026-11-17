#include "selftest_images.h"

#include <chrono>
#include <cstdio>
#include <map>
#include <set>
#include <thread>
#include <vector>

#include "../../src/featurize/json.h"
#include "card_images.h"
#include "files.h"
#include "image_renderer.h"
#include "paths.h"
#include "stb_image_write.h"
#include "table_model.h"

namespace {

const int PNG_W = 16, PNG_H = 22;

// One face of a synthetic manifest card: its name and file (relative to the cache), cached or not.
struct FakeFace {
        std::string name;
        std::string file;
        bool cached = true;
};
struct FakePrinting {
        std::string set, number, released;
        std::vector<FakeFace> faces;
};
struct FakeCard {
        std::string key;
        int index = -1;
        bool token = false;
        std::string stem;
        std::vector<std::string> names;
        int chosen = 0;
        std::vector<FakePrinting> printings;
};

}  // namespace

static uint32_t color_of(size_t i);
static bool write_png(const std::string &path, uint32_t rgba);
static std::string manifest_json(const std::vector<FakeCard> &cards, const std::string &missing_key);
static bool write_cache(const std::string &dir, const std::vector<FakeCard> &cards, const std::string &missing_key);
static std::vector<FakeCard> unit_cards();
static rmwork::CardFace face_of(int card, bool back, const std::string &name, bool token = false);
static bool wait_texture(rmwork::ImageTextures &t, const std::string &path, float w, ImTextureData **out);
static void check_manifest(const rmwork::CheckFn &check, const std::string &tmp);
static void check_resolution(const rmwork::CheckFn &check, const std::string &tmp);
static void check_lru(const rmwork::CheckFn &check);
static void check_textures(const rmwork::CheckFn &check, const std::string &tmp);
static bool draws_texture(const rmwork::CardRenderer &r, const rmwork::CardLook &look, const ImTextureData *tex);

// A distinct opaque colour per index.
static uint32_t color_of(size_t i) {
    const uint32_t r = static_cast<uint32_t>((i * 97 + 40) % 256), g = static_cast<uint32_t>((i * 57 + 90) % 256),
                   b = static_cast<uint32_t>((i * 31 + 160) % 256);
    return r | (g << 8) | (b << 16) | 0xFF000000u;
}

static bool write_png(const std::string &path, uint32_t rgba) {
    if (!rmwork::make_dirs(rmwork::parent_dir(path))) return false;
    std::vector<uint32_t> px(static_cast<size_t>(PNG_W * PNG_H), rgba);
    return stbi_write_png(path.c_str(), PNG_W, PNG_H, 4, px.data(), PNG_W * 4) != 0;
}

static std::string manifest_json(const std::vector<FakeCard> &cards, const std::string &missing_key) {
    using featurize::json_quote;
    std::string out = "{\"version\": 1, \"cards\": [";
    for (size_t c = 0; c < cards.size(); ++c) {
        const FakeCard &k = cards[c];
        out += std::string(c ? ", " : "") + "{\"key\": " + json_quote(k.key) +
               ", \"index\": " + std::to_string(k.index) + ", \"token\": " + (k.token ? "true" : "false") +
               ", \"stem\": " + json_quote(k.stem) + ", \"names\": [";
        for (size_t i = 0; i < k.names.size(); ++i) out += std::string(i ? ", " : "") + json_quote(k.names[i]);
        out += "], \"scryfall_name\": " + json_quote(k.names.empty() ? k.key : k.names[0]) +
               ", \"chosen\": " + std::to_string(k.chosen) + ", \"printings\": [";
        for (size_t p = 0; p < k.printings.size(); ++p) {
            const FakePrinting &pr = k.printings[p];
            out += std::string(p ? ", " : "") + "{\"set\": " + json_quote(pr.set) +
                   ", \"set_name\": " + json_quote("Set " + pr.set) + ", \"number\": " + json_quote(pr.number) +
                   ", \"released\": " + json_quote(pr.released) + ", \"faces\": [";
            for (size_t f = 0; f < pr.faces.size(); ++f)
                out += std::string(f ? ", " : "") + "{\"name\": " + json_quote(pr.faces[f].name) +
                       ", \"url\": \"\", \"file\": " + json_quote(pr.faces[f].file) + "}";
            out += "]}";
        }
        out += "]}";
    }
    out += "], \"missing\": [";
    if (!missing_key.empty()) out += "{\"key\": " + json_quote(missing_key) + ", \"why\": \"not on Scryfall\"}";
    return out + "]}\n";
}

static bool write_cache(const std::string &dir, const std::vector<FakeCard> &cards, const std::string &missing_key) {
    size_t n = 0;
    for (const FakeCard &c : cards)
        for (const FakePrinting &p : c.printings)
            for (const FakeFace &f : p.faces)
                if (f.cached && !write_png(dir + "/" + f.file, color_of(n++))) return false;
    std::string err;
    return rmwork::make_dirs(dir) &&
           rmwork::write_file_atomic(dir + "/manifest.json", manifest_json(cards, missing_key), &err);
}

// Lightning Bolt (two printings, the second not cached), Delver of Secrets (two faces), Grizzly
// Bears (not cached), the Clue token (by index) and Marit Lage (an unregistered token, by name).
static std::vector<FakeCard> unit_cards() {
    std::vector<FakeCard> cards(5);
    cards[0].key = "Lightning Bolt";
    cards[0].index = 2;
    cards[0].names = {"Lightning Bolt"};
    cards[0].printings = {{"lea", "161", "1993-08-05", {{"Lightning Bolt", "images/lea/161.png", true}}},
        {"m10", "146", "2009-07-17", {{"Lightning Bolt", "images/m10/146.png", false}}}};
    cards[1].key = "Delver of Secrets";
    cards[1].index = 16;
    cards[1].names = {"Delver of Secrets"};
    cards[1].printings = {{"isd", "51", "2011-09-30",
        {{"Delver of Secrets", "images/isd/51-0.png", true}, {"Insectile Aberration", "images/isd/51-1.png", true}}}};
    cards[2].key = "Grizzly Bears";
    cards[2].index = 3;
    cards[2].names = {"Grizzly Bears"};
    cards[2].printings = {{"lea", "190", "1993-08-05", {{"Grizzly Bears", "images/lea/190.png", false}}}};
    cards[3].key = "token:c_a_clue_draw";
    cards[3].index = 906;
    cards[3].token = true;
    cards[3].stem = "c_a_clue_draw";
    cards[3].names = {"Clue"};
    cards[3].printings = {{"tsoi", "11", "2016-04-08", {{"Clue", "images/tsoi/11.png", true}}}};
    cards[4].key = "token:marit_lage";
    cards[4].token = true;
    cards[4].stem = "marit_lage";
    cards[4].names = {"Marit Lage"};
    cards[4].printings = {{"v16", "16", "2016-08-19", {{"Marit Lage", "images/v16/16.png", true}}}};
    return cards;
}

static rmwork::CardFace face_of(int card, bool back, const std::string &name, bool token) {
    rmwork::CardFace f;
    f.card = card;
    f.back = back;
    f.name = name;
    f.token = token;
    return f;
}

// Hands frames to the textures until `path` is there (at most two seconds).
static bool wait_texture(rmwork::ImageTextures &t, const std::string &path, float w, ImTextureData **out) {
    for (int i = 0; i < 2000; ++i) {
        t.new_frame();
        if ((*out = t.get(path, w)) != nullptr) return true;
        if (t.failed(path)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

static void check_manifest(const rmwork::CheckFn &check, const std::string &tmp) {
    using namespace rmwork;
    ImageManifest m;
    std::string err;
    const std::vector<FakeCard> cards = unit_cards();
    check(m.parse(manifest_json(cards, "Air Elemental"), &err), "the synthetic manifest does not parse: " + err);
    check(m.cards().size() == 5, "the manifest's card count");
    check(m.missing().size() == 1 && m.missing()[0].first == "Air Elemental", "the manifest's missing list");
    const ImageCard *bolt = m.by_index(2);
    check(bolt && bolt->key == "Lightning Bolt" && bolt->printings.size() == 2 && bolt->chosen == 0,
        "Lightning Bolt by its vocab index");
    check(m.by_key("token:c_a_clue_draw") == m.by_index(906) && m.by_index(906), "a token by its index");
    check(m.token_by_name("marit lage token") && m.token_by_name("Marit Lage")->stem == "marit_lage",
        "an unregistered token by its name");
    check(m.by_index(-1) == nullptr, "a token without an index is listed by index");
    check(
        bolt && bolt->display() == "Lightning Bolt" && m.by_index(906)->display() == "Clue token", "the display names");
    check(!m.parse("{\"cards\": [", &err), "malformed JSON parses");
    check(!m.parse("{\"version\": 1}", &err), "a manifest without cards parses");
    std::vector<FakeCard> bad = cards;
    bad[0].printings[0].faces[0].file = "../../etc/passwd";
    check(!m.parse(manifest_json(bad, ""), &err), "a face file outside the cache is accepted");
    bad = cards;
    bad[0].printings[0].number.clear();
    check(!m.parse(manifest_json(bad, ""), &err), "a printing without a number is accepted");
    check(!m.load(tmp + "/no_such/manifest.json", &err) && err.empty(), "a missing manifest is an error");

    PrintingOverrides o{{"Lightning Bolt", {"m10", "146"}}, {"token:c_a_clue_draw", {"tsoi", "11"}}};
    PrintingOverrides back;
    check(printings_from_json(printings_to_json(o), &back, &err) && back == o, "the overrides round trip: " + err);
    check(printings_from_json("{}", &back, &err) && back.empty(), "empty overrides");
    check(
        !printings_from_json("{\"Lightning Bolt\": {\"set\": \"m10\"}}", &back, &err), "an override without a number");
    check(!printings_from_json("[1]", &back, &err), "overrides that are no object");
    const std::string path = printings_path(tmp + "/config");
    check(save_printings(path, o, &err) && load_printings(path, &back, &err) && back == o,
        "the overrides file round trip: " + err);
    check(load_printings(tmp + "/no_such/printings.json", &back, &err) && back.empty(), "a missing overrides file");
}

static void check_resolution(const rmwork::CheckFn &check, const std::string &tmp) {
    using namespace rmwork;
    const std::string cache = tmp + "/cache";
    check(write_cache(cache, unit_cards(), "Air Elemental"), "the synthetic cache cannot be written");
    CardImages images;
    const std::string overrides = tmp + "/config2/printings.json";
    images.configure(cache, overrides);
    check(images.has_manifest(), "the synthetic cache's manifest is not read: " + images.manifest_error());
    const CardFace bolt = face_of(2, false, "Lightning Bolt");
    check(images.file_for(bolt) == cache + "/images/lea/161.png", "Lightning Bolt's default image");
    const CardFace bears = face_of(3, false, "Grizzly Bears");
    check(images.file_for(bears).empty(), "an image not in the cache is drawn");
    check(images.file_for(face_of(16, false, "Delver of Secrets")) == cache + "/images/isd/51-0.png",
        "Delver of Secrets' front");
    check(images.file_for(face_of(16, true, "Insectile Aberration")) == cache + "/images/isd/51-1.png",
        "Delver of Secrets' back");
    check(images.file_for(face_of(16, true, "Some Other Name")) == cache + "/images/isd/51-1.png",
        "a back face of another name takes the second face");
    check(images.file_for(face_of(906, false, "Clue", true)) == cache + "/images/tsoi/11.png", "the Clue token");
    check(images.file_for(face_of(1023, false, "Marit Lage", true)) == cache + "/images/v16/16.png",
        "an unregistered token by name");
    check(images.file_for(face_of(1023, false, "Marit Lage", false)).empty(), "a card matched as a token by name");
    check(images.file_for(face_of(40, false, "Unknown Card")).empty(), "a card the manifest lacks has an image");
    ImageCacheStatus st = images.status();
    check(st.manifest && st.with_images == 2 && st.tokens == 2 && st.token_images == 2 && st.missing == 1 &&
              st.vocab > 300,
        "the cache's status: " + std::to_string(st.with_images) + " cards, " + std::to_string(st.token_images) +
            " tokens");

    std::string err;
    check(images.set_override("Lightning Bolt", {"m10", "146"}, &err), "an override is not saved: " + err);
    bool overridden = false;
    const ImageCard *card = images.manifest().by_key("Lightning Bolt");
    check(card && shown_printing(*card, images.overrides(), &overridden) == &card->printings[1] && overridden,
        "the override's printing is not shown");
    check(images.file_for(bolt).empty(), "an override not downloaded is drawn");
    check(images.status().with_images == 1, "the status counts an override not downloaded");
    check(
        card && fetch_printing_command(cache, *card, card->printings[1]) ==
                    "python3 tools/scryfall/fetch_images.py --cache " + cache + " --printing 'Lightning Bolt=m10:146'",
        "the fetch command");
    CardImages again;
    again.configure(cache, overrides);
    check(again.overrides().count("Lightning Bolt") && again.file_for(bolt).empty(), "the override is not kept");
    check(images.set_override("Lightning Bolt", {"xyz", "1"}, &err) &&
              images.file_for(bolt) == cache + "/images/lea/161.png",
        "an override the manifest does not list is not the default's");
    check(images.set_override("Lightning Bolt", PrintingChoice(), &err) && images.overrides().empty(),
        "the per-card reset");
    check(images.set_override("Delver of Secrets", {"isd", "51"}, &err) && images.clear_overrides(&err) &&
              images.overrides().empty(),
        "Reset All");
    again.reload();
    check(again.overrides().empty(), "Reset All is not written");

    ImageCardRenderer renderer(&images);
    CardLook look;
    check(!renderer.draws_image(look, 100), "a back is drawn as an image");
    look.face = &bears;
    check(!renderer.draws_image(look, 100), "a card without a cached image is drawn as an image");
    const CardFace front = face_of(16, false, "Delver of Secrets"),
                   backface = face_of(16, true, "Insectile Aberration");
    look.face = &front;
    look.other = &backface;
    check(renderer.other_face_differs(look), "a double-faced card's faces look the same");
    const CardFace split_a = face_of(2, false, "Lightning Bolt"), split_b = face_of(2, true, "Lightning Bolt");
    look.face = &split_a;
    look.other = &split_b;
    check(!renderer.other_face_differs(look), "one image is shown twice");

    Settings st_text;
    check(settings_from_json("{\"images\": {\"cache\": \"\", \"faces\": \"text\"}}", &st_text, &err) &&
              !st_text.card_images,
        "Card faces: text is not read: " + err);
    Settings round;
    check(settings_from_json(settings_to_json(st_text), &round, &err) && !round.card_images &&
              default_settings().card_images,
        "Card faces does not round-trip, or images is not the default");
    check(!settings_from_json("{\"images\": {\"faces\": \"pictures\"}}", &round, &err), "an unknown card face");

    CardImages none;
    none.configure(tmp + "/empty_cache", overrides);
    check(!none.has_manifest() && none.manifest_error().empty() && none.file_for(bolt).empty(),
        "no manifest: an image is drawn");
}

static void check_lru(const rmwork::CheckFn &check) {
    using namespace rmwork;
    TextureLru lru;
    lru.touch("a", 100, 1);
    lru.touch("b", 100, 2);
    lru.touch("c", 100, 3);
    check(lru.count() == 3 && lru.bytes() == 300, "the LRU's totals");
    check(lru.over_budget(10, 2, 1000) == std::vector<std::string>{"a"}, "the count budget evicts the oldest");
    check(lru.over_budget(10, 10, 150) == (std::vector<std::string>{"a", "b"}), "the byte budget");
    lru.touch("a", 100, 9);
    check(lru.over_budget(10, 1, 1000) == (std::vector<std::string>{"b", "c"}),
        "an entry drawn the frame before is evicted");
    check(lru.over_budget(4, 0, 0) == (std::vector<std::string>{"b"}), "an entry drawn this frame is evicted");
    lru.touch("b", 50, 2);
    check(lru.bytes() == 250, "a re-touched entry's bytes");
    lru.erase("b");
    lru.erase("nothing");
    check(lru.count() == 2 && lru.bytes() == 200 && !lru.contains("b"), "erase");
    check(texture_level(488, 300) == 0 && texture_level(488, 244) == 1 && texture_level(488, 100) == 2 &&
              texture_level(488, 10) == ImageTextures::MAX_LEVEL && texture_level(100, 300) == 0,
        "the texture levels");
    int w = 0, h = 0;
    const std::vector<uint32_t> in = {0xFF000000u, 0xFF000004u, 0xFF000008u, 0xFF00000Cu};
    const std::vector<uint32_t> out = halve_rgba(in, 2, 2, &w, &h);
    check(w == 1 && h == 1 && out.size() == 1 && out[0] == 0xFF000006u, "halving averages a 2 × 2 block");
    halve_rgba(std::vector<uint32_t>(15, 0), 3, 5, &w, &h);
    check(w == 1 && h == 2, "halving an odd size");
}

// Whether drawing `look` adds a draw command on `tex`.
static bool draws_texture(const rmwork::CardRenderer &r, const rmwork::CardLook &look, const ImTextureData *tex) {
    ImGui::NewFrame();
    ImDrawList *dl = ImGui::GetForegroundDrawList();
    r.draw(dl, look, ImVec2(10, 10), ImVec2(110, 149), true, 0.5f);
    bool found = false;
    for (const ImDrawCmd &c : dl->CmdBuffer) found = found || c.TexRef._TexData == tex;
    ImGui::EndFrame();
    return found;
}

static void check_textures(const rmwork::CheckFn &check, const std::string &tmp) {
    using namespace rmwork;
    const std::string cache = tmp + "/cache";
    CardImages images;
    images.configure(cache, tmp + "/config3/printings.json");
    ImageTextures &tx = images.textures();
    const std::string bolt = images.file_for(face_of(2, false, "Lightning Bolt"));
    check(tx.get(bolt, 300) == nullptr, "a texture is there before it is decoded");
    ImTextureData *tex = nullptr;
    check(wait_texture(tx, bolt, 300, &tex), "the decode thread hands no texture over");
    if (tex) {
        check(tex->Width == PNG_W && tex->Height == PNG_H && tex->Format == ImTextureFormat_RGBA32,
            "the texture's size and format");
        check(tex->Status == ImTextureStatus_WantCreate, "the texture is not handed to the backend");
        check(tex->Pixels && reinterpret_cast<const uint32_t *>(tex->Pixels)[0] == color_of(0),
            "the texture's pixels are not the image's");
    }
    ImTextureData *small = nullptr;
    check(wait_texture(tx, bolt, 2, &small) && small, "a smaller level");
    for (int i = 0; i < 2000 && small && small->Width == PNG_W; ++i) {
        tx.new_frame();
        small = tx.get(bolt, 2);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(small && small->Width == PNG_W / 8 && small->Height == PNG_H / 8, "a small card's level is not 1/8");
    const std::string broken = cache + "/images/broken.png";
    std::string err;
    write_file_atomic(broken, "not a png", &err);
    ImTextureData *none = nullptr;
    check(!wait_texture(tx, broken, 100, &none) && tx.failed(broken), "a file that does not decode");

    ImageCardRenderer renderer(&images);
    const CardFace bolt_face = face_of(2, false, "Lightning Bolt");
    CardLook look;
    look.face = &bolt_face;
    ImTextureData *at_draw = tx.get(bolt, 100);
    check(at_draw && draws_texture(renderer, look, at_draw), "the renderer does not draw the image");
    const CardFace bears = face_of(3, false, "Grizzly Bears");
    look.face = &bears;
    check(!draws_texture(renderer, look, at_draw), "a card without an image is drawn with another's");

    // Budget: one texture; the earlier ones go once they are not drawn.
    tx.set_budget(1, size_t(1) << 30);
    const std::string delver = images.file_for(face_of(16, false, "Delver of Secrets"));
    for (int i = 0; i < 4; ++i) tx.new_frame();
    ImTextureData *d = nullptr;
    check(wait_texture(tx, delver, 300, &d), "a second texture");
    for (int i = 0; i < 4; ++i) {
        tx.new_frame();
        tx.get(delver, 300);
    }
    check(tx.live() == 1 && tx.evicted() >= 2 && tx.lru().count() == 1, "textures past the budget are not evicted");
    check(tx.get(delver, 300) != nullptr, "the texture drawn is evicted");
    tx.clear();
    check(tx.live() == 0 && !tx.busy() && !tx.failed(broken), "clear");
}

namespace rmwork {

void check_image_units(const CheckFn &check, const std::string &tmp) {
    const std::string dir = tmp + "/images";
    make_dirs(dir);
    check_manifest(check, dir);
    check_resolution(check, dir);
    check_lru(check);
    check_textures(check, dir);
    remove_tree(dir);
}

void check_table_images(const CheckFn &check, App &app, const std::function<void(int)> &frames, const std::string &at) {
    ReplaySession &s = app.replay();
    if (!s.loaded()) return;
    const Match &m = *s.match();
    const CardBook &book = app.table().book(m, VIEW_OMNI);
    // Images for every other card id of the CARDS frame, both faces of a double-faced card.
    std::vector<FakeCard> cards;
    std::set<int> with;
    int n = 0;
    for (int id = 0; id < 1024; ++id) {
        const CardFace *front = book.face(id, false);
        if (!front || (n++ % 2)) continue;
        FakeCard c;
        c.key = front->token ? "token:t" + std::to_string(id) : front->name;
        c.index = id;
        c.token = front->token;
        c.names = {front->name};
        FakePrinting p{"tst", std::to_string(id), "2000-01-01", {}};
        p.faces.push_back({front->name, "images/tst/" + std::to_string(id) + "-0.png", true});
        if (const CardFace *back = book.face(id, true))
            p.faces.push_back({back->name, "images/tst/" + std::to_string(id) + "-1.png", true});
        c.printings.push_back(p);
        cards.push_back(c);
        with.insert(id);
    }
    const std::string cache = app.card_images().cache_dir();
    check(!cache.empty() && cache.find("rmwork_selftest") != std::string::npos,
        at + ": the image cache is not the test's: " + cache);
    if (cache.find("rmwork_selftest") == std::string::npos) return;
    check(write_cache(cache, cards, ""), at + ": the synthetic image cache cannot be written");
    app.card_images().reload();
    check(app.card_images().has_manifest(), at + ": the synthetic image cache is not read");

    ImageCardRenderer renderer(&app.card_images());
    int images = 0, texts = 0;
    const size_t step = std::max<size_t>(1, s.count() / 12);
    for (size_t i = 0; i < s.count(); i += step) {
        for (int v = 0; v < VIEW_COUNT; ++v) {
            s.go_to(i);
            s.set_view(v);
            frames(1);
            for (int k = 0; k < 400 && app.card_images().textures().busy(); ++k) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                frames(1);
            }
            frames(1);
            const TableModel *model = app.table().model();
            if (!model) continue;
            const CardBook &vb = app.table().book(m, v);
            for (const PlacedCard &pc : app.table().layout().cards) {
                const TableObject &o = model->objects[static_cast<size_t>(pc.object)];
                const CardFace *face = !o.face_down && o.card >= 0 ? vb.face(o.card, o.back) : nullptr;
                const std::string where = at + ": decision " + std::to_string(i + 1) + " view " + std::to_string(v) +
                                          ": " + (o.name.empty() ? std::string("a hidden card") : o.name);
                if (!face) {
                    CardLook back;
                    check(!renderer.draws_image(back, 100), where + ": a back is drawn as an image");
                    ++texts;
                    continue;
                }
                const std::string file = app.card_images().file_for(*face);
                const bool want = with.count(face->card) > 0;
                check(want == !file.empty(), where + (want ? ": no image" : ": an image the cache does not have"));
                if (file.empty()) {
                    ++texts;
                    continue;
                }
                check(!app.card_images().textures().failed(file), where + ": the image does not decode");
                CardLook look;
                look.face = face;
                const float w = pc.rotated ? pc.rect.h() : pc.rect.w();
                check(renderer.draws_image(look, w), where + ": the image is not drawn once decoded");
                ++images;
            }
        }
    }
    check(images > 0 && texts > 0,
        at + ": the Table drew " + std::to_string(images) + " images and " + std::to_string(texts) + " text cards");
    std::printf("rmwork: Table images (%s): %d cards drawn as images, %d as text\n", at.c_str(), images, texts);
    remove_tree(cache);
    app.card_images().reload();
    check(!app.card_images().has_manifest(), at + ": the synthetic image cache is still read");
}

}  // namespace rmwork
