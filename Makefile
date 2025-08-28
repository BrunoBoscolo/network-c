# Compiler and flags
CC = gcc
CFLAGS = -Iinclude -Wall -O3
LDFLAGS = -lm

# Source files and object files
SRCS = src/main.c src/neural_network.c src/evolution.c src/data_loader.c
OBJS = $(SRCS:.c=.o)

# Target executable
TARGET = main

# Test files
TEST_SRCS = test/test_runner.c test/test_matrix.c test/test_neural_network.c test/test_persistence.c test/test_evolution.c src/neural_network.c src/evolution.c
TEST_OBJS = $(TEST_SRCS:.c=.o)
TEST_TARGET = test_runner

# Recognizer files
RECOGNIZER_SRCS = src/number_recognizer.c src/neural_network.c src/evolution.c src/data_loader.c
RECOGNIZER_OBJS = $(RECOGNIZER_SRCS:.c=.o)
RECOGNIZER_TARGET = recognizer

# Default rule
all: $(TARGET)

# Rule for the main example
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Rule for the number recognizer
recognizer: $(RECOGNIZER_OBJS)
	$(CC) $(RECOGNIZER_OBJS) -o $(RECOGNIZER_TARGET) $(LDFLAGS)

# Rule to compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Test rule
test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJS)
	$(CC) $(TEST_OBJS) -o $(TEST_TARGET) $(LDFLAGS)

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET) $(TEST_OBJS) $(TEST_TARGET) $(RECOGNIZER_OBJS) $(RECOGNIZER_TARGET)

.PHONY: all clean test recognizer
