#ifndef BOARD_STATE_H
#define BOARD_STATE_H

#include "../ecs/system.h"

class BoardState : public System {
    public:
        void init();
        // called when anything resolves
        // trigger state based actions
        void update();
};

#endif /* BOARD_STATE_H */
