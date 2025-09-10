#include "debug.h"
#include "systems/orderer.h"
#include "ecs/coordinator.h"
#include "components/carddata.h"
#include "error.h"

void print_library(std::shared_ptr<Orderer> orderer, Zone::Ownership owner) {
    auto library = orderer->get_library_contents(owner);
    size_t n = library.size();

    for (size_t i = 0; i < n; i++){ 
        bool found = false;
        for (auto &&card_id : library){
            auto &card = global_coordinator.GetComponent<CardData>(card_id);
            auto &zone = global_coordinator.GetComponent<Zone>(card_id);
            if(zone.distance_from_top == i){
                printf("Card %s is %zu from top \n", card.uid.c_str(), i);
                found = true;
                break;
            }
        }
        if (!found) fatal_error("Missing card in library at index " + std::to_string(i));
    }
    
}
