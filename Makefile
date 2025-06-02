# Makefile for building align executable

CXX := g++
CXXFLAGS := -std=c++20 -O3 -Iinclude

TARGET := aligner
SRCS := src/main.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean
