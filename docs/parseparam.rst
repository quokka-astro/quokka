#This document includes the details on the all the parse-able params in Quokka.
#These parameters are read in the readParameters() function in simulation.hpp.


---------------------------------------------------------------------------—————————————————————————-
|  Param Name                      |       Type               |         Param Description            |
—————————————————————————----------------------------------------------------------------------------
max_timesteps                      |      Integer             | Maximum time steps for the simulation |
				   |			      |
cfl                                |      Float               | Set the cfl number for the simulation |
				   |			      |
amr_interpolation_method           |      String              | Sets how to interpolate between AMR levels??|
				   |			      |                                             |
stop_time                          |      Float               | Stop time of the simulation                 |
				   |			      |			                            |
ascent_interval                    | ???                      | ???                                         |
	                           |		              |       			                    |
plotfile_interval                  |      Integer             | The number of steps between plot file dumps |
			           |		 	      |                                             |
projection_interval                | ????                     | ???
  				   |			      |			                            |
statistics_interval                | ????                     |
				   |			      |			                            |
checkpointtime_interval            | ???                      |        ?????                                |
				   |			      |		                                                                                         |			                
checkpoint_interval                |       Integer            | The number of steps between successive checkpoint dumps                                          |
                                   |			      |			                                                                                 |
do_reflux                          |         ????             |                                                                                                  |
			           |			      |			                                                                                 |
suppress_output                    |         ????             |                  ?????                                                                           |
				   |                          |			                                                                                 |
derived_vars                       | ????                     | ????                                                                                             |
				   |			      |			                                                                                 |
regrid_interval                    |???                       | ?????                                                                                            |
			           |			      |			                                                                                 |
density_floor                      |        Float             | The floor on density values in the simulation. Enforced through EnforceLimits                    |
				   |			      |			                                                                                 |
temperature_ceiling                |        Float             | The ceiling on temperature values in the simulation. Enforced through EnforceLimits              |
				   |			      |                                                                                                  |
speed_ceiling                      |        Float             | The ceiling on the absolute value of speed in the simulation. Enforced through EnforceLimits.    |
 				   |			      |                                                                                                  |
max_walltime                       |        Float             | The duration of the simulation.                                                                  |

