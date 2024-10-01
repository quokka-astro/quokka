if gamma != 1: 
	cal T_gas 
	if dust_off:
		T_d = T_gas
	# cal T_d and dust_model # (0: no dust, 1: coupled dust, 2: decoupled dust) 
	cal is_dust_gas_decoupled, gamma_gd_time_dt 
	if dust_model == 0: # no dust
		while 1:
			if n == 0:
				R = chat * dt * (chi_B * b - chi_E * E_g) = tau_g * (b - X^-1 * E_g) 
			else:
				E_g = X * (b R / tau_g)
			def RHS(Egas, Egas0, Erad, Erad0, R, dt, rho):
				F_0 = Egas - Egas0 + cscale * sum(R) 
				F_g = Erad - Erad0 - R 
			F_0, F_g = RHS(Egas, Egas0, Erad, Erad0, R, dt, rho) 
			if F_0 == 0 and F_g == 0: 
				break
			def Jacobian(Egas, Erad, dt, rho):
				d_Eg_d_T = X * d_Bg_d_T 
				d_Eg_d_Rg = - X / tau_g 
				J00 = 1 
				J0g = cscale 
				Jg0 = 1/CV * d_Eg_d_T 
				Jgg = d_Eg_d_Rg - 1 
			jacobian = Jacobian(Egas, Erad, dt, rho) 
			delta_x, delta_R = SolveLinearEqs(jacobian)
			Egas += delta_x; R += delta_R
	elif dust_model == 1 # dust, coupled
		while 1:
			if n == 0:
				R = chat * dt * (chi_B * b - chi_E * E_g) = tau_g * (b - X^-1 * E_g) 
			else:
				T_d = T_gas - Lambda_gd / (Lambda_gd_0 * n * n * sqrt(T_gas)) 
				E_g = X * (b - R / tau_g)
			def RHS(Egas, Egas0, Erad, Erad0, R, dt, rho):
				d_Td_d_Rg = 1 / (N * sqrt(T_gas))
				d_Bg_d_R = d_Bg_d_T * d_Td_d_Rg
				rg = X * d_Bg_d_R
				F_0 = Egas - Egas0 + cscale * sum(R) 
				F_g = Erad - Erad0 - R - 1/cscale * rg * F_0
			F_0, F_g = RHS(Egas, Egas0, Erad, Erad0, R, dt, rho) 
		if F_0 == 0 and F_g == 0: 
			break
		def Jacobian(Egas, Erad, dt, rho):
			d_Td_d_T = 3. / 2. - T_d / (2. * T_gas) 
			d_Eg_d_T = X * d_Bg_d_T * d_Td_d_T 
			d_Eg_d_Rg = - X / tau_g 
			d_Td_d_Rg = -1 / (N * sqrt(T_gas))
			d_Bg_d_R = d_Bg_d_T * d_Td_d_Rg
			rg = X * d_Bg_d_R
			J00 = 1 
			J0g = cscale 
			Jg0 = 1/CV * d_Eg_d_T - 1/cscale * rg * J00
			Jgg = d_Eg_d_Rg - 1 
		jacobian = Jacobian(Egas, Erad, dt, rho) 
		delta_x, delta_R = SolveLinearEqs(jacobian)
		Egas += delta_x; R += delta_R
	else if dust_model == 2  # dust, decoupled
		while 1:
			if n == 0:
				R = chat * dt * (chi_B * b - chi_E * E_g) = tau_g * (b - X^-1 * E_g) 
			else:
				T_d = T_gas - Lambda_gd / (Lambda_gd_0 * n * n * sqrt(T_gas)) 
				E_g = X * (b - R / tau_g)
			def RHS(Egas, Egas0, Erad, Erad0, R, dt, rho):
				F_0 = -gamma_gd_time_dt + sum(R) 
				F_g = Erad - Erad0 - R
			F_0, F_g = RHS(Egas, Egas0, Erad, Erad0, R, dt, rho) 
			if F_0 == 0 and F_g == 0: 
				break
			def Jacobian(Egas, Erad, dt, rho):
				d_Eg_d_T = X * d_Bg_d_T
				d_Eg_d_Rg = - X / tau_g 
				J00 = 0 
				J0g = 1
				Jg0 = d_Eg_d_T
				Jgg = d_Eg_d_Rg - 1 
			jacobian = Jacobian(Egas, Erad, dt, rho) 
			delta_x, delta_R = SolveLinearEqs(jacobian)
			T_d += delta_x; R += delta_R
	elif dust_model == 3 # dust, coupled, + photoelectric heating
		while 1:
			if n == 0:
				R = chat * dt * (chi_B * b - chi_E * E_g) = tau_g * (b - X^-1 * E_g) 
			else:
				T_d = T_gas - Lambda_gd / (Lambda_gd_0 * n * n * sqrt(T_gas)) 
				E_g = X * (b - R / tau_g)
			def RHS(Egas, Egas0, Erad, Erad0, R, dt, rho):
				d_Td_d_Rg = 1 / (N * sqrt(T_gas))
				d_Bg_d_R = d_Bg_d_T * d_Td_d_Rg
				rg = X * d_Bg_d_R
				F_0 = Egas - Egas0 + cscale * sum(R) - PE_E1_deriv * Erad[1]
				F_g = Erad - Erad0 - R - (rg / cscale) * F_0
			F_0, F_g = RHS(Egas, Egas0, Erad, Erad0, R, dt, rho) 
		if F_0 == 0 and F_g == 0: 
			break
		def Jacobian(Egas, Erad, dt, rho):
			d_Td_d_T = 3. / 2. - T_d / (2. * T_gas) 
			d_Eg_d_T = X * d_Bg_d_T * d_Td_d_T 
			d_Eg_d_Rg = - X / tau_g 
			d_Td_d_Rg = -1 / (N * sqrt(T_gas))
			d_Bg_d_R = d_Bg_d_T * d_Td_d_Rg
			rg = X * d_Bg_d_R # this is Jg1, or X_g * P_g

			# original Jacobian
			J00 = 1
			J01 = cscale - PE_E1_deriv * d_Eg_d_Rg[1]
			J0g = cscale
			Jg0 = 1/CV * d_Eg_d_T
			Jgg = X * d_Bg_d_R - X / tau_g - 1
			Jgh = X * d_Bg_d_R # = rg

			# after elimination
			J00 = 1 
			J0g = cscale
			J01 = cscale - PE_E1_deriv * d_Eg_d_Rg[1]
			Jg0 = 1/CV * d_Eg_d_T - (rg / cscale) * J00
			Jg1 = rg - (rg / cscale) * J01 # USE THIS
			    = (rg / cscale) * PE_E1_deriv * d_Eg_d_Rg[1]
			Jgg = Jgg - (rg / cscale) * J0g = Jgg - rg = - X / tau_g - 1
			J11 = J11 - (rg / cscale) * J01 
			    = X * d_Bg_d_R - X / tau_g - 1 - (rg / cscale) * (cscale - PE_E1_deriv * d_Eg_d_Rg[1]) 
			    = - X / tau_g - 1 + (rg / cscale) * PE_E1_deriv * d_Eg_d_Rg[1] # USE THIS
		jacobian = Jacobian(Egas, Erad, dt, rho) 
		delta_x, delta_R = SolveLinearEqs(jacobian)
		Egas += delta_x; R += delta_R
	else if dust_model == 4  # dust, decoupled, + photoelectric heating
		while 1:
			if n == 0:
				R = chat * dt * (chi_B * b - chi_E * E_g) = tau_g * (b - X^-1 * E_g) 
			else:
				T_d = T_gas - Lambda_gd / (Lambda_gd_0 * n * n * sqrt(T_gas)) 
				E_g = X * (b - R / tau_g)
			def RHS(Egas, Egas0, Erad, Erad0, R, dt, rho):
				F_0 = -gamma_gd_time_dt + sum(R) 
				F_g = Erad - Erad0 - R
			F_0, F_g = RHS(Egas, Egas0, Erad, Erad0, R, dt, rho) 
			if F_0 == 0 and F_g == 0: 
				break
			def Jacobian(Egas, Erad, dt, rho):
				d_Eg_d_T = X * d_Bg_d_T
				d_Eg_d_Rg = - X / tau_g 
				J00 = 0 
				J0g = 1
				Jg0 = d_Eg_d_T
				Jgg = d_Eg_d_Rg - 1 
			jacobian = Jacobian(Egas, Erad, dt, rho) 
			delta_x, delta_R = SolveLinearEqs(jacobian)
			T_d += delta_x; R += delta_R