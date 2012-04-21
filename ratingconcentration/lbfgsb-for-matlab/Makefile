# Mac OS X settings.
MEX         = /Applications/MATLAB_R2010b.app/bin/mex
MEXSUFFIX   = mexmaci64
MATLAB_HOME = /Applications/MATLAB_R2010b.app/
CXX         = gcc
F77         = gfortran
CFLAGS      = -O3 -fPIC -fno-common -fexceptions -no-cpp-precomp 
FFLAGS      = -O3 -x f77-cpp-input -fPIC -fno-common -m64

TARGET = lbfgsb.$(MEXSUFFIX)
OBJS   = solver.o matlabexception.o matlabscalar.o matlabstring.o   \
         matlabmatrix.o arrayofmatrices.o program.o matlabprogram.o \
         lbfgsb.o

CFLAGS += -Wall -ansi -DMATLAB_MEXFILE

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CFLAGS) -I$(MATLAB_HOME)/extern/include -o $@ -c $^

%.o: %.f
	$(F77) $(FFLAGS) -o $@ -c $^

$(TARGET): $(OBJS)
	$(MEX) -cxx CXX=$(CXX) CC=$(CXX) FC=$(FCC) LD=$(CXX) -lgfortran -lm \
        -O -output $@ $^

clean:
	rm -f *.o $(TARGET)

