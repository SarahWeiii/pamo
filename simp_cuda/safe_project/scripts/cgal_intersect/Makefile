CXX = g++
CXXFLAGS =
LDFLAGS = -lmpfr -lgmp

SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = cgal_intersect

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)