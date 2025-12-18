# Makefile for sourceTerm StirFromFile (implementation level)

Stir += Stir_data.o st_calcAccel.o st_read_modes_file.o

Stir_init.o : Stir_data.o st_read_modes_file.o Driver_data.o \
	      Driver_interface.o RuntimeParameters_interface.o Grid_data.o
Stir.o : Stir_data.o st_read_modes_file.o Driver_data.o IO_data.o \
         Grid_interface.o Driver_interface.o Timers_interface.o
Stir_computeDt.o : Stir_data.o Driver_interface.o Driver_data.o Grid_interface.o
Stir_finalize.o : Stir_data.o
st_read_modes_file.o : Stir_data.o Driver_data.o
st_calcAccel.o : Stir_data.o Grid_interface.o
