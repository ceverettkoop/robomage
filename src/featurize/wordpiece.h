#ifndef FEATURIZE_WORDPIECE_H
#define FEATURIZE_WORDPIECE_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ── The BERT WordPiece tokenizer (plan step 5c, docs/featurizer.md §5), standard library only ──
//
// The reference is Hugging Face's BertTokenizerFast over a model's vocab.txt and
// tokenizer_config.json; tools/embed/make_goldens.py records its ids for every text entry, and the
// gate (vocab tier) checks this tokenizer against them.
//
//   0. The special tokens ([CLS], [SEP], [UNK], [PAD], [MASK], as tokenizer_config.json names them)
//      are matched in the raw text first and become their ids; the text between them goes through
//      the steps below.
//   1. Clean: drop NUL, U+FFFD and control and format characters (tab, newline and carriage return
//      excepted); every whitespace character becomes a space.
//   2. CJK ideographs (the unified blocks) are split off as words of their own
//      (tokenize_chinese_chars).
//   3. An uncased model (do_lower_case, strip_accents null or true) lowercases and strips accents
//      (NFD, then every nonspacing mark dropped); a cased one (do_lower_case false, strip_accents
//      null or false) keeps the text. Other combinations are refused at load.
//   4. Split on whitespace; every punctuation character (ASCII punctuation and Unicode category P)
//      is a word of its own.
//   5. WordPiece per word: greedy longest match from the left, continuations prefixed "##"; a word
//      longer than 100 characters, or one with a part no vocab entry matches, is [UNK].
//   6. [CLS] ids… [SEP].
//
// Steps 1-4 need Unicode properties; this tokenizer carries them for ASCII, Latin-1 Supplement,
// Latin Extended-A, the combining diacritical marks, General Punctuation, Mathematical Operators,
// the ideographic space and the unified CJK blocks. A text with any other character is refused
// (encode returns false naming it) rather than tokenized on a guess: extend the tables (and the
// golden's coverage strings, tools/embed/make_goldens.py) first.

namespace featurize {

class WordPiece {
    public:
        // Reads `model_dir`/vocab.txt (one token per line, the line number its id) and
        // `model_dir`/tokenizer_config.json. False with `err` when either is unreadable, the vocab
        // lacks a special token, or the configuration is one this tokenizer does not implement.
        bool load(const std::string &model_dir, std::string *err);

        // The ids of `text` (UTF-8): [CLS], the WordPiece ids, [SEP]; never truncated. False with
        // `err` on malformed UTF-8 or a character outside the tables above.
        bool encode(const std::string &text, std::vector<int64_t> *ids, std::string *err) const;

        size_t vocab_size() const { return tokens_.size(); }
        int64_t pad_id() const { return pad_; }

    private:
        bool segment_ids(const std::string &text, std::vector<int64_t> *ids, std::string *err) const;
        bool word_ids(const std::vector<uint32_t> &word, std::vector<int64_t> *ids) const;
        bool normalise(const std::vector<uint32_t> &in, std::vector<uint32_t> *out, std::string *err) const;

        std::vector<std::string> tokens_;
        std::unordered_map<std::string, int64_t> ids_;
        std::vector<std::pair<std::string, int64_t>> specials_;
        bool uncased_ = true;
        bool chinese_ = true;
        int64_t cls_ = -1, sep_ = -1, unk_ = -1, pad_ = -1;
};

}  // namespace featurize

#endif /* FEATURIZE_WORDPIECE_H */
