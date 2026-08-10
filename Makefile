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
# Auto-generated files kept in sync with the C++ sources at build time
# (train/ codegen plus the C++ mirror header src/gen/card_costs_gen.h).
PYGEN := train/_enums.py train/card_costs.py train/card_props.py \
	src/gen/card_costs_gen.h src/gen/archetypes_gen.h
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

ifeq ($(BUILD),RELEASE)
	CFLAGS += -O2 -flto -march=native -DNDEBUG
	CXXFLAGS += -O2 -flto -march=native -DNDEBUG
else
	CFLAGS += $(DEBUGFLAGS) $(CHECKFLAGS) $(C_CHECKFLAGS)
	CXXFLAGS += $(DEBUGFLAGS) $(CHECKFLAGS)
	LDFLAGS += -rdynamic
endif

C_SRCS := $(wildcard $(SRCDIR)/*.c)
C_SRCS += $(wildcard $(SRCDIR)/*/*.c)
CXX_SRCS := $(wildcard $(SRCDIR)/*.cpp)
CXX_SRCS += $(wildcard $(SRCDIR)/*/*.cpp)
# The actor (bin/az_actor) is a SEPARATE binary that links libtorch and needs
# exceptions on. Keep its TUs out of the engine build entirely so plain `make`,
# `make check`, and bin/robomage never see torch or an exceptions-enabled TU.
CXX_SRCS := $(filter-out $(SRCDIR)/actor/%,$(CXX_SRCS))
_C_OBJ := $(patsubst $(SRCDIR)/%.c,%.o,$(C_SRCS))
_CXX_OBJ += $(patsubst $(SRCDIR)/%.cpp,%.o,$(CXX_SRCS))
C_OBJ = $(patsubst %,$(ODIR)/%,$(_C_OBJ))
CXX_OBJ = $(patsubst %,$(ODIR)/%,$(_CXX_OBJ))
DEPS = $(C_OBJ:.o=.d) $(CXX_OBJ:.o=.d)

$(ODIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(IFLAGS) $(CFLAGS) $(DEPFLAGS) $(PLATFLAGS)

# NOTE: this actor-specific pattern rule MUST stay defined before the generic
# $(ODIR)/%.o: $(SRCDIR)/%.cpp rule below. Apple's bundled GNU Make (3.81, the
# last GPLv2 release — macOS never ships GNU Make 4.x) resolves competing
# pattern rules by definition order rather than by shortest/most-specific stem
# as newer make and the GNU Make manual describe. If the generic rule comes
# first, `make actor` on macOS silently compiles src/actor/*.cpp with the
# engine's IFLAGS/CXXFLAGS instead of ACTOR_IFLAGS/ACTOR_CXXFLAGS, dropping
# -I$(SRCDIR) (breaking `#include "classes/..."`) and leaving -fno-exceptions on.
$(ODIR)/actor/%.o: $(SRCDIR)/actor/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -c -o $@ $< $(ACTOR_IFLAGS) $(ACTOR_CXXFLAGS) $(DEPFLAGS) $(PLATFLAGS)

$(ODIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -c -o $@ $< $(IFLAGS) $(CXXFLAGS) $(DEPFLAGS) $(PLATFLAGS)

.DEFAULT_GOAL := all

all: pygen program

# Regenerate the train/ codegen files from the C++ sources. Each is a real file
# target with its inputs as prerequisites, so codegen only runs when those
# sources actually change (not on every build).
pygen: $(PYGEN)

train/_enums.py: src/classes/action.h src/classes/game.h src/classes/gamestate.h src/machine_io.h train/gen_enums.py
	$(PYTHON) train/gen_enums.py

# One generator run emits BOTH the Python matrices and the C++ mirror header, so
# they are grouped targets (`&:`, GNU Make 4.3+): the recipe runs once and is
# guaranteed to have produced every listed target — parallel-safe, and never
# invoked twice under `make -j`.
train/card_costs.py src/gen/card_costs_gen.h &: src/card_vocab.h src/machine_io.h train/gen_card_costs.py
	$(PYTHON) train/gen_card_costs.py

# Frozen per-card printed-property block (network-side only; no C++ mirror).
# gen_card_costs.py and train/_enums.py are prerequisites because the generator
# imports helpers from one and _OBS_KEYWORDS from the other; card scripts stay
# out of the prerequisites, same convention as card_costs.
train/card_props.py: src/card_vocab.h src/machine_io.h train/gen_card_props.py train/gen_card_costs.py train/_enums.py
	$(PYTHON) train/gen_card_props.py

# C++ mirror of the deck -> archetype -> value-bucket metadata, so the in-process
# AZ actor rebuilds the same observation matchup tail train/env.py appends.
src/gen/archetypes_gen.h: train/archetypes.py bin/resources/decks/archetypes.json train/gen_archetypes.py
	$(PYTHON) train/gen_archetypes.py

program:$(C_OBJ) $(CXX_OBJ)
	@mkdir -p $(BINDIR)
	$(CXX) -o $(BINDIR)/$(BINNAME) $(C_OBJ) $(CXX_OBJ) $(LDFLAGS) $(LDLIBS) $(PLATFLAGS)

# Standardized engine test gate — the single command CI and developers run. Builds
# first (via `all`), then runs every tier of train/ci_check.py (codegen-sync, vocab
# coverage, byte-identical replay corpus, deterministic league smoke, short fuzz).
# Exits nonzero on any finding. Requires the card scripts to be provisioned first
# (tools/forge_fetch/provision_decks.py; the SessionStart hook does this).
check: all
	$(PYTHON) train/ci_check.py

# Regenerate every derived artifact after an intentional C++ change: provision
# the card set (codegen reads card scripts), force-rerun every generator, and
# re-record the replay corpus with the fresh binary. Run this when `make check`
# reports stale codegen or corpus drift, then commit the results. The generators
# run unconditionally here (not via the incremental pygen file targets) because
# ci_check's pygen tier restores the committed copies with `git checkout`, which
# bumps their mtimes past the C++ headers and turns the mtime-based `pygen`
# target into a silent no-op on stale content.
regen: all
	$(PYTHON) tools/forge_fetch/provision_decks.py
	$(PYTHON) train/gen_enums.py
	$(PYTHON) train/gen_card_costs.py
	$(PYTHON) train/gen_card_props.py
	$(PYTHON) train/gen_archetypes.py
	$(PYTHON) train/regression/replay_diff.py record

# ── bin/az_actor — in-process AlphaZero actor (Phase D), NOT in the default build ──
# Links all engine objects EXCEPT obj/main.o (the actor provides its own main),
# plus the src/actor/* TUs (compiled with exceptions ON) and libtorch. Auto-detect
# libtorch from the project venv; override with `make actor LIBTORCH_DIR=/path`.
LIBTORCH_DIR ?= $(shell ls -d train/.venv/lib/python3*/site-packages/torch 2>/dev/null | head -1)

# Fail early (with a clear message) only when actually asked to build the actor.
ifneq (,$(filter actor,$(MAKECMDGOALS)))
ifeq ($(strip $(LIBTORCH_DIR)),)
$(error LIBTORCH_DIR is empty: could not find train/.venv/.../site-packages/torch. \
Install torch into the venv, or pass LIBTORCH_DIR=/path/to/torch)
endif
endif

ACTOR_SRCS := $(wildcard $(SRCDIR)/actor/*.cpp)
ACTOR_OBJ := $(patsubst $(SRCDIR)/%.cpp,$(ODIR)/%.o,$(ACTOR_SRCS))
ENGINE_OBJ_NO_MAIN := $(filter-out $(ODIR)/main.o,$(C_OBJ) $(CXX_OBJ))

# torch headers are noisy under -Wconversion etc.; -isystem silences them. ABI is
# 1 in this venv (matches the engine's default), so no -D_GLIBCXX_USE_CXX11_ABI.
TORCH_INCLUDES := -isystem $(LIBTORCH_DIR)/include -isystem $(LIBTORCH_DIR)/include/torch/csrc/api/include
# Same base flags as the engine MINUS -fno-exceptions (torch throws), PLUS torch
# includes and -I$(SRCDIR) (actor TUs use non-relative engine includes).
ACTOR_CXXFLAGS := $(filter-out -fno-exceptions,$(CXXFLAGS)) $(TORCH_INCLUDES)
ACTOR_IFLAGS := $(IFLAGS) -I$(SRCDIR)

actor: pygen $(ENGINE_OBJ_NO_MAIN) $(ACTOR_OBJ)
	@mkdir -p $(BINDIR)
	$(CXX) -o $(BINDIR)/az_actor $(ENGINE_OBJ_NO_MAIN) $(ACTOR_OBJ) \
		$(LDFLAGS) $(LDLIBS) $(PLATFLAGS) \
		-L$(LIBTORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(abspath $(LIBTORCH_DIR)/lib)

# ── actor-syntax — libtorch-free compile check of the actor's obs layout mirror ──
# obs_builder.{h,cpp} carries the actor's copy of the observation layout, pinned by
# a wall of static_asserts against the engine's constants. Those asserts are the
# only thing that catches the C++ actor drifting from src/machine_io.h — but they
# only fire under `make actor`, which needs libtorch and is not in the default
# build, so a layout change can (and did) land with the actor left uncompilable.
# These TUs are the actor's only torch-free ones, so -fsyntax-only fires every
# layout assert with nothing but a compiler. Wired into ci_check.py's `actorobs`
# tier, which IS part of `make check`. td_targets.cpp joins them for the same
# reason: it is the C++ twin of az_selfplay.py's n-step TD rule and must stay
# compilable wherever libtorch is not installed.
ACTOR_SYNTAX_SRCS := $(SRCDIR)/actor/obs_builder.cpp $(SRCDIR)/actor/npz_writer.cpp \
                     $(SRCDIR)/actor/td_targets.cpp

actor-syntax: pygen
	@for f in $(ACTOR_SYNTAX_SRCS); do \
		echo "  syntax-only: $$f"; \
		$(CXX) -fsyntax-only $$f $(ACTOR_IFLAGS) \
			$(filter-out -fno-exceptions,$(CXXFLAGS)) $(PLATFLAGS) || exit 1; \
	done

.PHONY: all pygen check regen clean actor actor-syntax

# Remove everything under the object tree (the compile rules mkdir -p subdirs
# back on demand), not per-level globs that silently miss deeper nesting, plus
# the linked binary so a failed build can't leave a stale bin/robomage for the
# harness/CI to use. $(ODIR)/* (not $(ODIR)) keeps the git-tracked
# obj/.gitphony placeholder; bin/ is kept — it holds resources/ and decks.
clean:
	rm -rf $(ODIR)/*
	rm -f $(BINDIR)/$(BINNAME)

-include $(DEPS)
# The actor TUs are filtered out of CXX_SRCS (they only build under `make actor`),
# so their generated dep files must be included separately — otherwise editing a
# header they share with the engine (e.g. obs_builder.h) leaves stale actor
# objects behind and bin/az_actor links two different obs layouts together.
-include $(ACTOR_OBJ:.o=.d)