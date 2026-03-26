LDFLAGS = -shared
TARGET_LIB = ../libFahes.so
CHILD_MAKEFILE_DIR = FAHES_Code/src
ARGS = -C $(CHILD_MAKEFILE_DIR) TARGET=$(TARGET_LIB) LFLAGS+=$(LDFLAGS)

.PHONY: all
all:
	$(MAKE) $(ARGS) $@

.PHONY: clean
clean:
	$(MAKE) $(ARGS) clean $@
rmo:
	$(MAKE) $(ARGS) rmo $@
