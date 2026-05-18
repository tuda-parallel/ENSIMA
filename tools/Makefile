CC = gcc
CFLAGS = -Wall -O3

all : csv_parser

csv_parser: Import-CSV.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f csv_parser