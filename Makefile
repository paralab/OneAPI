CXX = dpcpp
CXXFLAGS = -g -O3 -o
LDFLAGS = -lOpenCL -lsycl
EXE_NAME = benchOneAPI
SOURCES = src/benchOneAPI.cpp
BINDIR = bin

DPCPP_CXX = dpcpp
DPCPP_CXXFLAGS = -g -O3 -o
MKL_CXXFLAGS = -I$(MKLROOT)/include -DMKL_ILP64
MKL_LDFLAGS = ${MKLROOT}/lib/intel64/libmkl_sycl.a  -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lOpenCL -ldl

all: benchOneAPI vdot mvec matmat

benchOneAPI:
	[ -d $(BINDIR) ] || mkdir $(BINDIR)
	$(CXX) $(CXXFLAGS) $(BINDIR)/$(EXE_NAME) $(SOURCES) $(LDFLAGS)

clean: 
	rm -rf $(BINDIR)/*



MKL_EXE_NAME_CD = cdot
MKL_SOURCES_CD = src/vecdot.cpp

vdot:
	$(DPCPP_CXX) $(MKL_CXXFLAGS) $(DPCPP_CXXFLAGS) $(BINDIR)/$(MKL_EXE_NAME_CD) $(MKL_SOURCES_CD) $(MKL_LDFLAGS)



MKL_EXE_NAME_MV = mvec
MKL_SOURCES_MV = src/mvec.cpp

mvec:
	$(DPCPP_CXX) $(MKL_CXXFLAGS) $(DPCPP_CXXFLAGS) $(BINDIR)/$(MKL_EXE_NAME_MV) $(MKL_SOURCES_MV) $(MKL_LDFLAGS)

MKL_EXE_NAME_MM = matmat
MKL_SOURCES_MM = src/matmat.cpp

matmat:
	$(DPCPP_CXX) $(MKL_CXXFLAGS) $(DPCPP_CXXFLAGS) $(BINDIR)/$(MKL_EXE_NAME_MM) $(MKL_SOURCES_MM) $(MKL_LDFLAGS)
