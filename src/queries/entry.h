#ifndef QUERIES_ENTRY_H
#define QUERIES_ENTRY_H

#include "../ecs/entity.h"

struct CardData;

// The face a card will have on the battlefield as it enters (CR 614.12 / 712.14a): its back face
// when it is entering transformed (EntryInfo::enters_transformed, a modal back face played or
// a card returned transformed) or is already a transformed permanent, else its front. Replacement
// effects that check what an entering object "would be" read this face.
const CardData &entering_face(Entity e, const CardData &cd);

// What is recorded about `e`'s next battlefield entry (components/entry_info.h). entry_info adds
// the component when `e` has none; find_entry_info returns null then.
struct EntryInfo;
EntryInfo &entry_info(Entity e);
EntryInfo *find_entry_info(Entity e);
// The entry `e` was headed for isn't happening (it went elsewhere, its move onto the battlefield
// was prevented, its cast was reversed): forget everything recorded about it.
void drop_entry_info(Entity e);
// Remove `e`'s EntryInfo once every fact in it has been consumed.
void drop_entry_info_if_consumed(Entity e);

#endif /* QUERIES_ENTRY_H */
