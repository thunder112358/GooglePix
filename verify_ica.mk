CC = gcc
CFLAGS = -Wall -O2 -g
LDFLAGS = -lm

# Source files for verification
VERIFY_SRCS = ica.c verify_ica.c block_matching.c
VERIFY_OBJS = $(VERIFY_SRCS:.c=.o)
VERIFY_TARGET = verify_ica

# Main verification target
verify: $(VERIFY_TARGET)

$(VERIFY_TARGET): $(VERIFY_OBJS)
	$(CC) $(VERIFY_OBJS) -o $(VERIFY_TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean verification files
clean_verify:
	rm -f $(VERIFY_OBJS) $(VERIFY_TARGET)

.PHONY: verify clean_verify