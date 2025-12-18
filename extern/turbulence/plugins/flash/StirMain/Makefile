# Makefile for sourceTerm Stir (implementation level)

Stir += Stir_data.o st_stir_TurbGen_interface.o

Stir_init.o : Stir_data.o Driver_data.o Driver_interface.o \
	RuntimeParameters_interface.o st_stir_TurbGen_interface.o
Stir.o : Stir_data.o Driver_interface.o Driver_data.o Grid_interface.o \
	Timers_interface.o st_stir_TurbGen_interface.o
Stir_computeDt.o : Stir_data.o Driver_interface.o Driver_data.o Grid_interface.o
st_stir_TurbGen_interface.o : TurbGen.h
