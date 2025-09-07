#ifndef OWNER_H
#define OWNER_H

const bool PLAYER_A = 0;
const bool PLAYER_B = 1;

struct Owner{
    Owner(bool in_value){value = in_value;};
    bool value;
};

#endif /* OWNER_H */
