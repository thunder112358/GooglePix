CC = gcc
CFLAGS = -Wall -Wextra -O3 -ffast-math -march=native
LDFLAGS = -lm

# Source files
SRCS = main.c block_matching.c ica.c utils.c
OBJS = $(SRCS:.c=.o)
TARGET = image_align

# Header files
HEADERS = block_matching.h ica.h utils.h

# External dependencies
DEPS = stb_image.h stb_image_write.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

# Download stb dependencies if they don't exist
$(DEPS):
	@echo "Downloading $@..."
	@curl -s -o $@ https://raw.githubusercontent.com/nothings/stb/master/$@

# Make sure we have the dependencies before building
main.o: $(DEPS) 