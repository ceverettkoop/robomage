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

# Per-configuration output trees. CONFIG (set in the BUILD if/else below) selects
# debug vs release so the two builds never clobber each other's objects or binary:
#   obj/debug/…   bin/debug/robomage      (plain `make`)
#   obj/release/… bin/release/robomage    (`make BUILD=RELEASE`)
# This also removes the old "must `make clean` when switching BUILD" hazard: object
# files are keyed on source path alone, so a config switch used to relink stale
# objects compiled with the wrong flags. Separate trees give each config its own.
ODIR=obj/$(CONFIG)
SRCDIR=src
DEPFLAGS = -MMD -MP
BINDIR=bin/$(CONFIG)
BINNAME=robomage

# Python for card-script provisioning and the corpus recorder (stdlib only).
PYTHON ?= python3
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
	CONFIG := release
	CFLAGS += -O2 -flto -march=native -DNDEBUG
	CXXFLAGS += -O2 -flto -march=native -DNDEBUG
else
	CONFIG := debug
	CFLAGS += $(DEBUGFLAGS) $(CHECKFLAGS) $(C_CHECKFLAGS)
	CXXFLAGS += $(DEBUGFLAGS) $(CHECKFLAGS)
	LDFLAGS += -rdynamic
endif

C_SRCS := $(wildcard $(SRCDIR)/*.c)
C_SRCS += $(wildcard $(SRCDIR)/*/*.c)
CXX_SRCS := $(wildcard $(SRCDIR)/*.cpp)
CXX_SRCS += $(wildcard $(SRCDIR)/*/*.cpp)
# The actor TUs (src/actor/, `make actor`) link libtorch and need exceptions on.
# Keep them out of the engine build entirely so plain `make`,
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

all: program rmknow robomage_check

program:$(C_OBJ) $(CXX_OBJ)
	@mkdir -p $(BINDIR)
	$(CXX) -o $(BINDIR)/$(BINNAME) $(C_OBJ) $(CXX_OBJ) $(LDFLAGS) $(LDLIBS) $(PLATFLAGS)

# Standardized engine test gate — the single command CI and developers run. Builds first (via
# `all`), then runs every tier of the C++ gate bin/<config>/robomage_check (docs/ci.md: vocab,
# concede, rules scenarios, byte-identical replay corpus, state-format and knowledge self-checks,
# explore fuzz, bench, and the smoke tier through the Python driver). Exits nonzero on any finding.
# Requires the card scripts to be provisioned first (tools/forge_fetch/provision_decks.py; the
# SessionStart hook does this).
check: all
	$(BINDIR)/robomage_check

# Re-record the replay corpus after an intentional behavior change: provision the card set, build,
# then play the scripted agent's corpus games with the fresh binary (each game's decision log and,
# through robomage_check --tier replay --record, its replay baseline; python/record_corpus.py). Run
# this when `make check` reports corpus drift, then commit the results.
regen:
	$(PYTHON) tools/forge_fetch/provision_decks.py
	$(MAKE) all
	$(PYTHON) python/record_corpus.py

# Force a clean re-fetch of every Forge card/token script, then VALIDATE the fresh
# scripts against the COMMITTED replay corpus. Deletes the gitignored local scripts
# (bin/resources/cardsfolder + tokenscripts) so provision_decks.py re-fetches them
# all at the pinned Forge commit (tools/forge_fetch/FORGE_PIN) instead of skipping
# already-present files, then runs the standard `make check` gate — which builds and
# diffs the result against the committed corpus.
# Unlike `regen`, it deliberately does NOT re-record the corpus: the point is to
# DETECT whether the refetched scripts drift from the committed baseline. A corpus
# failure here means an upstream script change moved engine
# behavior — investigate it; if the change is intentional, run `make regen` to
# re-record and commit the new corpus.
refetch:
	rm -f bin/resources/cardsfolder/*/*.txt bin/resources/tokenscripts/*.txt
	$(PYTHON) tools/forge_fetch/provision_decks.py
	$(MAKE) check

# ── make actor — the libtorch TUs of src/actor/, NOT in the default build ──
# src/actor/ holds the TorchScript evaluator (az_evaluator.cpp), compiled with exceptions ON
# against libtorch. There is no actor binary until plan step 5 adds a main: `make actor` compiles
# the actor objects, and that binary will link $(ENGINE_OBJ_NO_MAIN) $(ACTOR_OBJ) $(ACTOR_LDFLAGS).
# Auto-detect libtorch from the torch package $(PYTHON) imports (an active venv's included), only
# when asked to build the actor; override with `make actor LIBTORCH_DIR=/path`. Fail early (with a
# clear message) when there is none.
ifneq (,$(filter actor,$(MAKECMDGOALS)))
LIBTORCH_DIR ?= $(shell $(PYTHON) -c 'import os, torch; print(os.path.dirname(torch.__file__))' 2>/dev/null)
ifeq ($(strip $(LIBTORCH_DIR)),)
$(error LIBTORCH_DIR is empty: $(PYTHON) cannot import torch. \
Install torch for it, or pass LIBTORCH_DIR=/path/to/torch)
endif
endif

ACTOR_SRCS := $(wildcard $(SRCDIR)/actor/*.cpp)
ACTOR_OBJ := $(patsubst $(SRCDIR)/%.cpp,$(ODIR)/%.o,$(ACTOR_SRCS))
ENGINE_OBJ_NO_MAIN := $(filter-out $(ODIR)/main.o,$(C_OBJ) $(CXX_OBJ))

# torch headers are noisy under -Wconversion etc.; -isystem silences them. ABI is
# 1 in the pip torch builds (matches the engine's default), so no -D_GLIBCXX_USE_CXX11_ABI.
TORCH_INCLUDES := -isystem $(LIBTORCH_DIR)/include -isystem $(LIBTORCH_DIR)/include/torch/csrc/api/include
# Same base flags as the engine MINUS -fno-exceptions (torch throws), PLUS torch
# includes and -I$(SRCDIR) (actor TUs use non-relative engine includes).
ACTOR_CXXFLAGS := $(filter-out -fno-exceptions,$(CXXFLAGS)) $(TORCH_INCLUDES)
ACTOR_IFLAGS := $(IFLAGS) -I$(SRCDIR)

# When the torch is a ROCm build (libtorch_hip.so present), link the HIP
# backend too so a `--device cuda` evaluator can reach the Radeon. The libs must be
# wrapped in --no-as-needed: nothing references their symbols directly (the
# backend registers itself via static initializers), so without it the linker
# drops them and the device silently never exists — the same gotcha as CUDA's
# torch_cuda. CPU-only torch builds link without them.
ACTOR_TORCH_LIBS := -ltorch -ltorch_cpu -lc10
ifneq (,$(wildcard $(LIBTORCH_DIR)/lib/libtorch_hip.so))
ACTOR_TORCH_LIBS += -Wl,--no-as-needed -ltorch_hip -lc10_hip -Wl,--as-needed
endif
ACTOR_LDFLAGS := $(LDFLAGS) $(LDLIBS) $(PLATFLAGS) \
                 -L$(LIBTORCH_DIR)/lib $(ACTOR_TORCH_LIBS) -Wl,-rpath,$(abspath $(LIBTORCH_DIR)/lib)

actor: $(ACTOR_OBJ)

# ── bin/<config>/rmknow — the knowledge sidecar over a recorded RMSTATE stream ──
# tools/rmknow/ (outside src/, so its main stays out of the engine) linked with ONLY the std-only
# objects it may see: the RMSTATE decoder and schema and src/knowledge/ (step-1 handoff §3). Part
# of the default build; `make rmknow` builds it alone.
RMKNOW_SRCS := $(wildcard tools/rmknow/*.cpp)
RMKNOW_OBJ := $(patsubst tools/%.cpp,$(ODIR)/tools/%.o,$(RMKNOW_SRCS))
RMKNOW_LIB_OBJ := $(ODIR)/state_format/decode.o $(ODIR)/state_format/schema.o \
                  $(patsubst $(SRCDIR)/%.cpp,$(ODIR)/%.o,$(wildcard $(SRCDIR)/knowledge/*.cpp))

$(ODIR)/tools/%.o: tools/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -c -o $@ $< $(IFLAGS) $(CXXFLAGS) $(DEPFLAGS) $(PLATFLAGS)

rmknow: $(RMKNOW_OBJ) $(RMKNOW_LIB_OBJ)
	@mkdir -p $(BINDIR)
	$(CXX) -o $(BINDIR)/rmknow $(RMKNOW_OBJ) $(RMKNOW_LIB_OBJ) $(LDFLAGS) $(LDLIBS) $(PLATFLAGS)

# ── bin/<config>/robomage_check — the C++ test gate (docs/ci.md, plan step 3a) ──
# tools/robomage_check/ runs the engine as subprocesses and links NO engine translation unit: only
# the std-only RMSTATE decoder and schema and the agents' std-only frame reader and Python-compatible
# generator. Built with the engine's flags (-fno-exceptions); `make check` runs every tier.
RMCHECK_SRCS := $(wildcard tools/robomage_check/*.cpp)
RMCHECK_OBJ := $(patsubst tools/%.cpp,$(ODIR)/tools/%.o,$(RMCHECK_SRCS))
RMCHECK_LIB_OBJ := $(ODIR)/state_format/decode.o $(ODIR)/state_format/schema.o \
                   $(ODIR)/agents/seat_frames.o $(ODIR)/agents/py_random.o

robomage_check: $(RMCHECK_OBJ) $(RMCHECK_LIB_OBJ)
	@mkdir -p $(BINDIR)
	$(CXX) -o $(BINDIR)/robomage_check $(RMCHECK_OBJ) $(RMCHECK_LIB_OBJ) $(LDFLAGS) $(LDLIBS) -pthread $(PLATFLAGS)

.PHONY: all check regen refetch clean actor rmknow robomage_check

# Remove BOTH configs' object trees and engine binaries (not just the current
# CONFIG's), so a plain `make clean` is config-independent — the compile rules
# mkdir -p the per-config subdirs back on demand. `obj/*` (glob skips dotfiles)
# keeps the git-tracked obj/.gitphony placeholder while removing obj/debug and
# obj/release; `bin/*/robomage` removes both configs' linked binaries so a failed
# build can't leave a stale one for the harness/CI. bin/ itself is kept — it holds
# resources/ and decks.
clean:
	rm -rf obj/*
	rm -f bin/*/$(BINNAME) bin/*/rmknow bin/*/robomage_check

-include $(DEPS)
# The actor TUs are filtered out of CXX_SRCS (they only build under `make actor`),
# so their generated dep files must be included separately — otherwise editing a
# header they share with the engine leaves stale actor objects behind.
-include $(ACTOR_OBJ:.o=.d)
-include $(RMKNOW_OBJ:.o=.d)
-include $(RMCHECK_OBJ:.o=.d)