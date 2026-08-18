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

# Python used to regenerate the train/ codegen files. Prefer the project venv;
# fall back to python3 (both generators use only the stdlib).
PYTHON := $(shell [ -x train/.venv/bin/python ] && echo train/.venv/bin/python || echo python3)
# Auto-generated codegen, regenerated on every build by the `pygen` target below.
# Tracked (derive only from tracked C++/JSON sources): train/_enums.py,
# src/gen/archetypes_gen.h. Untracked (derive from gitignored/fetched card-script
# content, so regenerated locally): train/card_costs.py, train/card_props.py,
# src/gen/card_costs_gen.h.
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

# Regenerate ALL codegen on EVERY build. card_costs.py / card_props.py and the C++
# mirror header src/gen/card_costs_gen.h derive from card-script CONTENT (ManaCost,
# keywords, types), and card scripts are gitignored/fetched — so their content is NOT
# expressible as a Make prerequisite. The old mtime-based file targets went stale
# whenever a script changed without a header/vocab change (e.g. a FORGE_PIN bump adding
# a keyword), which is exactly how a stale card_props.py got committed. Running the
# generators unconditionally here closes that hole; they write-if-changed
# (train/gen_util.py), so an unchanged output keeps its mtime and forces no recompile.
# stdout is muted (real failures raise and exit nonzero); errors still surface on stderr.
pygen:
	@$(PYTHON) train/gen_enums.py >/dev/null
	@$(PYTHON) train/gen_card_costs.py >/dev/null
	@$(PYTHON) train/gen_card_props.py >/dev/null
	@$(PYTHON) train/gen_archetypes.py >/dev/null

# The generated C++ mirror headers (src/gen/*.h) are #included by the engine, and
# src/gen/card_costs_gen.h is UNTRACKED — so guarantee pygen has produced them before any
# TU compiles (a fresh clone has no header and no dep files yet). Order-only (|): a mere
# regeneration never forces a rebuild; a real header content change still rebuilds its
# dependent objects through the -MMD/-include dep files.
$(C_OBJ) $(CXX_OBJ): | pygen

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
# the card set (codegen reads card scripts), then re-record the replay corpus with
# the fresh binary. Run this when `make check` reports corpus drift, then commit the
# results. `all` already regenerates all codegen unconditionally (the `pygen` target),
# so provisioning first — which may fetch changed scripts — guarantees the codegen it
# produces reflects the current card set before the corpus is recorded against it.
regen:
	$(PYTHON) tools/forge_fetch/provision_decks.py
	$(MAKE) all
	$(PYTHON) train/regression/replay_diff.py record

# Force a clean re-fetch of every Forge card/token script, then VALIDATE the fresh
# scripts against the COMMITTED replay corpus. Deletes the gitignored local scripts
# (bin/resources/cardsfolder + tokenscripts) so provision_decks.py re-fetches them
# all at the pinned Forge commit (tools/forge_fetch/FORGE_PIN) instead of skipping
# already-present files, then runs the standard `make check` gate — which
# regenerates all codegen (card_costs.py / card_props.py / src/gen/card_costs_gen.h)
# from the fresh scripts, builds, and diffs the result against the committed corpus.
# Unlike `regen`, it deliberately does NOT re-record the corpus: the point is to
# DETECT whether the refetched scripts drift from the committed baseline. A corpus
# (or codegen-sync) failure here means an upstream script change moved engine
# behavior — investigate it; if the change is intentional, run `make regen` to
# re-record and commit the new corpus.
refetch:
	rm -f bin/resources/cardsfolder/*/*.txt bin/resources/tokenscripts/*.txt
	$(PYTHON) tools/forge_fetch/provision_decks.py
	$(MAKE) check

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

# Same order-only pygen guard as the engine objects: the actor TUs include the
# untracked generated headers too, so pygen must produce them before they compile.
$(ACTOR_OBJ): | pygen

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
                     $(SRCDIR)/actor/td_targets.cpp $(SRCDIR)/actor/oracle_client.cpp

actor-syntax: pygen
	@for f in $(ACTOR_SYNTAX_SRCS); do \
		echo "  syntax-only: $$f"; \
		$(CXX) -fsyntax-only $$f $(ACTOR_IFLAGS) \
			$(filter-out -fno-exceptions,$(CXXFLAGS)) $(PLATFLAGS) || exit 1; \
	done

.PHONY: all pygen check regen refetch clean actor actor-syntax

# Remove BOTH configs' object trees and engine binaries (not just the current
# CONFIG's), so a plain `make clean` is config-independent — the compile rules
# mkdir -p the per-config subdirs back on demand. `obj/*` (glob skips dotfiles)
# keeps the git-tracked obj/.gitphony placeholder while removing obj/debug and
# obj/release; `bin/*/robomage` removes both configs' linked binaries so a failed
# build can't leave a stale one for the harness/CI. bin/ itself is kept — it holds
# resources/ and decks — and bin/*/az_actor is left in place (it is expensive to
# relink against libtorch and is gitignored separately).
clean:
	rm -rf obj/*
	rm -f bin/*/$(BINNAME)

-include $(DEPS)
# The actor TUs are filtered out of CXX_SRCS (they only build under `make actor`),
# so their generated dep files must be included separately — otherwise editing a
# header they share with the engine (e.g. obs_builder.h) leaves stale actor
# objects behind and bin/az_actor links two different obs layouts together.
-include $(ACTOR_OBJ:.o=.d)