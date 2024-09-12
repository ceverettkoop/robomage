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

ODIR=obj
SRCDIR=src
BINDIR=bin
BINNAME=robomage
GUI=false

CXXFLAGS = -ggdb
CFLAGS = -ggdb
IFLAGS = -Iinclude -Icomponents -isystem
LDFLAGS = -L./lib
LDLIBS =
CHECKFLAGS = -Wall -Wformat -Wformat=2 -Wconversion -Wimplicit-fallthrough \
-D_GLIBCXX_ASSERTIONS \
-fstack-protector-strong
C_CHECKFLAGS = -Werror=implicit -Werror=incompatible-pointer-types -Werror=int-conversion

ifeq ($(BUILD),RELEASE)
	CFLAGS += -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3 -o2
	CXXFLAGS += += -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3 -o2
else
	CFLAGS += $(CHECKFLAGS)
	CFLAGS += $(C_CHECKFLAGS)
	CXXFLAGS +=$(CHECKFLAGS)
endif

ifeq ($(GUI),TRUE)
	LDLIBS += -lraylib
	ifeq ($(PLATFORM),OSX)
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

$(ODIR)/%.o: $(SRCDIR)/%.c
	$(CC) -c -o $@ $< $(IFLAGS) $(CFLAGS) $(PLATFLAGS)

$(ODIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) -c -o $@ $< $(IFLAGS) $(CXXFLAGS) $(PLATFLAGS)

program:$(C_OBJ) $(CXX_OBJ)
	$(CXX) -o $(BINDIR)/$(BINNAME) $(C_OBJ) $(CXX_OBJ) $(LDFLAGS) $(LDLIBS) $(PLATFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*/*.o
	rm -f $(ODIR)/*.o