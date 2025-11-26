TARGET_EXEC := neural_network

BUILD_DIR := ./build
SRC_DIRS := ./src
INC_DIR := ./include

SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c')
INCLDS := $(shell find $(INC_DIR) -name '*.hpp' -or -name '*.h')

OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

INC_DIRS := $(shell find $(INC_DIR) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# Default build (no ASan)
$(TARGET_EXEC): $(OBJS)
	$(CXX) -g $(OBJS) -o $@ $(LDFLAGS)

# Compile object files
$(BUILD_DIR)/%.o: % $(INCLDS)
	mkdir -p $(dir $@)
	$(CXX) -O3 -DNDEBUG $(INC_FLAGS) $(CXXFLAGS) -c $< -o $@

# AddressSanitizer build
.PHONY: asan
asan: clean $(OBJS)
	@echo "Building with AddressSanitizer..."
	$(CXX) -fsanitize=address -g $(OBJS) -o $(TARGET_EXEC) $(LDFLAGS)

# Clean
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET_EXEC)
	rm -f out.perf-folded flamegraph.svg perf.data

# Flame graph generation
.PHONY: flame
flame: $(TARGET_EXEC)
	@echo "Generating flame graph..."
	perf record -F 99 -g ./$(TARGET_EXEC)
	perf report -n --stdio
	#perf script | ./FlameGraph/stackcollapse-perf.pl > out.perf-folded
	#FlameGraph/flamegraph.pl out.perf-folded > flamegraph.svg
	#@echo "Flame graph generated: flamegraph.svg"

