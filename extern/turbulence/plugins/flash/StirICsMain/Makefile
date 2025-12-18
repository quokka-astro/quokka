# Makefile for sourceTerm StirICs (Implementation level)

Stir += StirICs_data.o st_stirics_TurbGen_interface.o

StirICs_init.o : StirICs_data.o Driver_data.o RuntimeParameters_interface.o
StirICs.o : StirICs_data.o Driver_data.o Driver_interface.o Timers_interface.o Grid_interface.o \
            Grid_data.o PhysicalConstants_interface.o st_stirics_TurbGen_interface.o
st_stirics_TurbGen_interface.o : TurbGen.h
