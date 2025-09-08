# Compiler and flags
CC = gcc
CFLAGS = -Iinclude -Wall -O3 -fPIC
LDFLAGS = -lm

# --- Library ---
LIB_NAME = gann
LIB_SRCS = lib/gann_errors.c lib/matrix.c lib/data_loader.c lib/evolution.c lib/neural_network.c lib/gann.c lib/backpropagation.c lib/gann_backprop.c lib/selection.c lib/crossover.c lib/mutation.c
LIB_OBJS = $(LIB_SRCS:.c=.o)
STATIC_LIB = lib$(LIB_NAME).a
SHARED_LIB = lib$(LIB_NAME).so

# --- Examples ---
EXAMPLE_BINS = examples/training examples/recognizer examples/activations_comparison examples/backprop_training examples/comparison examples/ex_tournament_selection examples/ex_uniform_crossover examples/ex_arithmetic_crossover examples/ex_non_uniform_mutation examples/ex_adaptive_mutation

# --- Tests ---
TEST_SRCS = test/test_runner.c test/test_matrix.c test/test_neural_network.c test/test_persistence.c test/test_evolution.c test/test_backpropagation.c test/test_optimizers.c test/test_genetic_operators.c test/test_data_loader.c test/test_gann_errors.c
TEST_OBJS = $(TEST_SRCS:.c=.o)
TEST_TARGET = test_runner

# --- Targets ---

# Default rule: build libraries and examples
all: libs examples

# Rule to build both static and shared libraries
libs: $(STATIC_LIB) $(SHARED_LIB)

# Rule to build the examples
examples: $(EXAMPLE_BINS)

# Rule to build the static library
$(STATIC_LIB): $(LIB_OBJS)
	ar rcs $@ $^

# Rule to build the shared library
$(SHARED_LIB): $(LIB_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

# Rules for building examples
examples/training: examples/training.c $(STATIC_LIB)
	$(CC) $(CFLAGS) $< -o $@ $(STATIC_LIB) $(LDFLAGS)

examples/recognizer: examples/recognizer.c $(STATIC_LIB)
	$(CC) $(CFLAGS) $< -o $@ $(STATIC_LIB) $(LDFLAGS)

examples/activations_comparison: examples/activations_comparison.c $(STATIC_LIB)
	$(CC) $(CFLAGS) $< -o $@ $(STATIC_LIB) $(LDFLAGS)

examples/backprop_training: examples/backprop_training.c $(STATIC_LIB)
	$(CC) $(CFLAGS) $< -o $@ $(STATIC_LIB) $(LDFLAGS)

examples/comparison: examples/comparison.c $(STATIC_LIB)
	$(CC) $(CFLAGS) $< -o $@ $(STATIC_LIB) $(LDFLAGS)

examples/ex_tournament_selection: examples/ex_tournament_selection.c $(STATIC_LIB)
	$(CC) $(CFLAGS) $< -o $@ $(STATIC_LIB) $(LDFLAGS)

examples/ex_uniform_crossover: examples/ex_uniform_crossover.c $(STATIC_LIB)
	$(CC) $(CFLAGS) $< -o $@ $(STATIC_LIB) $(LDFLAGS)

examples/ex_arithmetic_crossover: examples/ex_arithmetic_crossover.c $(STATIC_LIB)
	$(CC) $(CFLAGS) $< -o $@ $(STATIC_LIB) $(LDFLAGS)

examples/ex_non_uniform_mutation: examples/ex_non_uniform_mutation.c $(STATIC_LIB)
	$(CC) $(CFLAGS) $< -o $@ $(STATIC_LIB) $(LDFLAGS)

examples/ex_adaptive_mutation: examples/ex_adaptive_mutation.c $(STATIC_LIB)
	$(CC) $(CFLAGS) $< -o $@ $(STATIC_LIB) $(LDFLAGS)

# Rule to compile library source files into object files
lib/%.o: lib/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile test source files into object files
test/%.o: test/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Test rule
test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJS) $(STATIC_LIB)
	$(CC) $(CFLAGS) $(TEST_OBJS) -o $@ $(STATIC_LIB) $(LDFLAGS)

# Clean rule
clean:
	rm -f lib/*.o $(STATIC_LIB) $(SHARED_LIB)
	rm -f $(EXAMPLE_BINS)
	rm -f test/*.o $(TEST_TARGET)

.PHONY: all clean test libs examples
