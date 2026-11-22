#ifndef RMWORK_TOKENS_VIEW_H
#define RMWORK_TOKENS_VIEW_H

#include <memory>
#include <string>
#include <vector>

#include "../../src/featurize/perturb.h"
#include "../../src/featurize/setup.h"
#include "attribution.h"
#include "tool_data.h"

// ── The Tokens tool: the decision's tokens by slots, labelled by the spec (docs/workbench_plan.md §3) ─
//
// One seat's frame featurized (featurize_traced with the inputs the Replay evaluator loaded: the seat
// view's own frame, or in the omniscient view the decider's), shown one token type at a time: a row per
// token (named by its frame row, perturb.h's state label), a column per slot of the type (named by its
// source column, the spec's position in the tooltip) holding the slot's values. With an attribution of
// the same frame (attribution.h) the cells it evaluated are its heat layer, coloured by the current
// target, and the tokens with heat come first. A footer counts the tokens, relations and menu rows. Without the
// featurizer's inputs (no evaluation through the server yet this session) it says so.

namespace rmwork {

struct TokensContext {
        std::shared_ptr<const featurize::Featurization> features;  // the Replay evaluator's, or null
        ToolData *data = nullptr;
        int seat = -1;                     // the seat whose frame is featurized
        const Attribution *att = nullptr;  // the attribution, when it is of this decision and seat
        const AttrTints *tints = nullptr;
};

class TokensView {
    public:
        void draw(const TokensContext &c);
        // The token type shown (-1: the one with the most tokens), and the tokens and their slots as last
        // drawn (the checks read them).
        int type() const { return type_; }
        void set_type(int t) { type_ = t; }
        // Screenshots: show the token type named `name` once it is present ("" off).
        void force_type(const std::string &name) { force_type_ = name; }
        const featurize::Tensors &tensors() const { return tensors_; }
        const featurize::UnitList &units() const { return list_; }
        // The cells drawn with a heat value at the last draw.
        size_t heat_cells() const { return heat_cells_; }

    private:
        bool refresh(const TokensContext &c, std::string *why);
        std::string cell_text(const featurize::SlotDef &s, size_t tok) const;

        int type_ = -1;
        std::string force_type_;
        std::string key_;  // what was featurized: decision, seat, the inputs, the attribution's tensors
        featurize::Tensors tensors_;
        featurize::UnitList list_;
        std::vector<std::vector<featurize::SlotDef>> slots_;
        std::shared_ptr<const featurize::Featurization> features_;
        size_t heat_cells_ = 0;
};

}  // namespace rmwork

#endif /* RMWORK_TOKENS_VIEW_H */
