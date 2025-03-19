CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -Iinclude
LDFLAGS = -lm

SRCS = src/simple_mnist.c src/mnist_loader.c src/neon_ops.c
OBJS = $(SRCS:.c=.o)
TARGET = simple_mnist

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
