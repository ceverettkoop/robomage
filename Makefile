PLATFORM =
ifeq ($(OS),Windows_NT)
	PLATFORM=WIN32
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		PLATFORM = LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		PLATFORM = OSX
	endif
endif

CC=gcc
CXX=g++

ifeq ($(PLATFORM), OSX)
	CC=clang
	CXX=clang++
endif

ODIR=obj
SRCDIR=src
DEPFLAGS = -MMD -MP
BINDIR=bin
BINNAME=robomage

# Python used to regenerate the train/ codegen files. Prefer the project venv;
# fall back to python3 (both generators use only the stdlib).
PYTHON := $(shell [ -x train/.venv/bin/python ] && echo train/.venv/bin/python || echo python3)
# Auto-generated Python files kept in sync with the C++ sources at build time.
PYGEN := train/_enums.py train/card_costs.py
GUI=TRUE
HEADLESS:=FALSE
DEBUGFLAGS = -ggdb
CXXFLAGS = -std=c++17 -fno-exceptions
CFLAGS =
IFLAGS = -Iinclude -Icomponents
LDFLAGS = -L./lib -fno-exceptions
LDLIBS =
CHECKFLAGS = -Wall -Wformat -Wformat=2 -Wconversion -Wimplicit-fallthrough \
-D_GLIBCXX_ASSERTIONS \
-fstack-protector-strong
C_CHECKFLAGS = -Werror=implicit -Werror=incompatible-pointer-types -Werror=int-conversion -Wno-sign-conversion -Wno-conversion

ifeq ($(HEADLESS), TRUE)
	GUI=FALSE
endif

ifeq ($(GUI),TRUE)
	CXXFLAGS += -DGUI=TRUE
	CFLAGS += -DGUI=TRUE
endif

ifeq ($(BUILD),RELEASE)
	CFLAGS += -O2 -flto -march=native -DNDEBUG
	CXXFLAGS += -O2 -flto -march=native -DNDEBUG
else
	CFLAGS += $(DEBUGFLAGS) $(CHECKFLAGS) $(C_CHECKFLAGS)
	CXXFLAGS += $(DEBUGFLAGS) $(CHECKFLAGS)
	LDFLAGS += -rdynamic
endif

ifeq ($(GUI),TRUE)
	LDLIBS += -lraylib
	ifeq ($(PLATFORM),OSX)
		IFLAGS += -I`brew --prefix raylib`/include
		LDFLAGS += -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL
	endif
endif

C_SRCS := $(wildcard $(SRCDIR)/*.c)
C_SRCS += $(wildcard $(SRCDIR)/*/*.c)
CXX_SRCS := $(wildcard $(SRCDIR)/*.cpp)
CXX_SRCS += $(wildcard $(SRCDIR)/*/*.cpp)
_C_OBJ := $(patsubst $(SRCDIR)/%.c,%.o,$(C_SRCS))
_CXX_OBJ += $(patsubst $(SRCDIR)/%.cpp,%.o,$(CXX_SRCS))
C_OBJ = $(patsubst %,$(ODIR)/%,$(_C_OBJ))
CXX_OBJ = $(patsubst %,$(ODIR)/%,$(_CXX_OBJ))
DEPS = $(C_OBJ:.o=.d) $(CXX_OBJ:.o=.d)

$(ODIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(IFLAGS) $(CFLAGS) $(DEPFLAGS) $(PLATFLAGS)

$(ODIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -c -o $@ $< $(IFLAGS) $(CXXFLAGS) $(DEPFLAGS) $(PLATFLAGS)

.DEFAULT_GOAL := all

all: pygen program

# Regenerate the train/ codegen files from the C++ sources. Each is a real file
# target with its inputs as prerequisites, so codegen only runs when those
# sources actually change (not on every build).
pygen: $(PYGEN)

train/_enums.py: src/classes/action.h src/classes/game.h train/gen_enums.py
	$(PYTHON) train/gen_enums.py

train/card_costs.py: src/card_vocab.h src/machine_io.h train/gen_card_costs.py
	$(PYTHON) train/gen_card_costs.py

program:$(C_OBJ) $(CXX_OBJ)
	@mkdir -p $(BINDIR)
	$(CXX) -o $(BINDIR)/$(BINNAME) $(C_OBJ) $(CXX_OBJ) $(LDFLAGS) $(LDLIBS) $(PLATFLAGS)

.PHONY: all pygen clean

clean:
	rm -f $(ODIR)/*/*.o $(ODIR)/*/*.d
	rm -f $(ODIR)/*.o $(ODIR)/*.d

-include $(DEPS)