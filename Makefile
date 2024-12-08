CC = gcc
CFLAGS = -Wall -O2 -g
LDFLAGS = -lm

SRCS = block_matching.c verify_block_matching.c
OBJS = $(SRCS:.c=.o)
TARGET = verify_block_matching

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)