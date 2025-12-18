#ifndef PARTICLE_RADIATION_HPP_
#define PARTICLE_RADIATION_HPP_

#include "AMReX_Extension.H"
#include "fundamental_constants.H"
#include "particle_types.hpp"
#include "util/DataTable.hpp"

namespace quokka
{

constexpr amrex::Real seconds_per_year = 3.15576e+07;
 
#if AMREX_SPACEDIM == 3

// GPU-friendly const table access for luminosity tables
// Nout should match nGroups in the problem
template <int Nout = 1, quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp> struct LuminosityGpuConstTables {
	quokka::DataTableGpuConst<2, Nout, oob_policy> luminosity; // 2D table: (age, mass) -> luminosity per group
};

// Host-side luminosity table storage
template <int Nout = 1, quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp> class LuminosityTables
{
      public:
	quokka::DataTable<2, Nout, oob_policy> luminosity; // 2D table: (age, mass) -> luminosity per group

	[[nodiscard]] auto const_tables() const -> LuminosityGpuConstTables<Nout, oob_policy>
	{
		LuminosityGpuConstTables<Nout, oob_policy> tables{luminosity.const_tables()};
		return tables;
	}

	[[nodiscard]] auto is_initialized() const -> bool { return luminosity.is_initialized(); }
};

// Static pointer to the current simulation's luminosity tables (set during initialization)
// Default to single output (Nout=1) and clamp out-of-bounds for backward compatibility
template <int Nout = 1, quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp>
inline LuminosityTables<Nout, oob_policy> *g_luminosity_tables_ptr = nullptr; // NOLINT

// Class to handle luminosity updates for stellar particles
class LuminosityUpdate
{
      public:
	template <typename problem_t, typename ParticleType, int Nout, quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateLuminosity(ParticleType &p, amrex::Real current_time,
									 LuminosityGpuConstTables<Nout, oob_policy> const &gpu_tables) noexcept
	{
		constexpr int nGroups = Physics_Traits<problem_t>::nGroups;
		static_assert(nGroups == Nout, "Number of groups must match table outputs");

		// Use table interpolation: (age, mass) -> luminosity per group
		const int mass_idx = StochasticStellarPopParticleMassIdx;
		const int birth_time_idx = StochasticStellarPopParticleBirthTimeIdx;
		const int lum_idx = StochasticStellarPopParticleLumIdx;
		const amrex::Real age_in_seconds = current_time - p.rdata(birth_time_idx);
		const amrex::Real mass = p.rdata(mass_idx);

		const amrex::Real mass_in_solar_masses = mass / C::M_solar;
		amrex::Real age_in_years = age_in_seconds / seconds_per_year;
		age_in_years = std::max(age_in_years, 1.0e-30); // age = 0 is allowed
		// Table coordinates: (age, mass) as specified in CSV input_names
		std::array<amrex::Real, 2> const point = {age_in_years, mass_in_solar_masses};

		// Interpolate luminosity from table (returns array with nGroups elements)
		// Conversion from log space is handled automatically by DataTable::interpolate()
		auto const luminosities = gpu_tables.luminosity.interpolate(point);

		// Update luminosity components (they are stored consecutively starting at lum_idx)
		if (lum_idx + nGroups <= ParticleType::NReal) {
			for (int g = 0; g < nGroups; ++g) {
				p.rdata(lum_idx + g) = luminosities[g];
			}
		}
	}
};

// Class to handle luminosity updates for protostellar particles
class ProtoLuminosityUpdate
{
      public:
	template <typename problem_t, typename ParticleType, int Nout, quokka::OutOfBounds oob_policy = quokka::OutOfBounds::clamp>
	AMREX_GPU_DEVICE AMREX_FORCE_INLINE static void updateProtoLuminosity(ParticleType &p, amrex::Real current_time,
									 LuminosityGpuConstTables<Nout, oob_policy> const &gpu_tables) noexcept
	{
	        const amrex::Real NAVOG = 6.022e23;   
	        const amrex::Real ERGEV = 1.6e-12;    // Number of ergs per eV
	        const amrex::Real FACC = 0.5; // Fraction of accreted energy that comes out as radiation, rather than being advected into the stellar interior or used to drive a wind
		const amrex::Real FK = 0.5;     // Fraction of energy the falls into the sink particle but is radiated away from the the inner disk before reaching the stellar surface
	        const amrex::Real FWIND = 0.21; // Fraction of accreted mass ejected in a wind
		const amrex::Real FRAD = 0.33;  // A radiative barrier forms when L_deuterium <= FRAD * L_ZAMS. See McKee & Tan 2002
		const amrex::Real SHELLFAC = 2.1;  // Radius increases by SHELLFAC when shell burning starts
	        const amrex::Real THAY = 3000.0;     // Hayashi temperature
		const amrex::Real TDEUT = 1.5e6;  // Temperature when deuterium burning starts
		const amrex::Real PSIION = (16.0*ERGEV*NAVOG);  // Energy per gram needed to dissociate and ionize a molecular gas with solar abundances
		const amrex::Real PSID = (100*ERGEV*NAVOG);  // Energy per gram released by burning the deuterium in a gas with solar abundances
		const amrex::Real MSUN = 1.989e33;    // Solar mass
		const amrex::Real RSUN = 6.96e10;    // Solar mass
		const amrex::Real MRADMIN = (0.01*MSUN);  // Minimum mass at which we use the model
		  
		// Use table interpolation: (age, mass) -> luminosity per group
		const int mass_idx = StarParticleMassIdx;
		const int birth_time_idx = StarParticleBirthTimeIdx;
		const int lum_idx = StarParticleLumIdx;
		// const amrex::Real age_in_seconds = current_time - p.rdata(birth_time_idx);
		const amrex::Real mass = p.rdata(mass_idx);
		const amrex::Real mlast = p.rdata(mlast_idx);
		const amrex::Real mdeut = p.rdata(mdeut_idx) + mass - mlast;
		const amrex::Real npoly = p.rdata(npoly_idx);
		const amrex::Real mdot = (mass - mlast) / dt;
		const amrex::Real burnState = p.rdata(burnState_idx);
		const amrex::Real l_hist = p.rdata(l_hist_idx);
		const amrex::Real msol = mass / C::M_solar;
		// amrex::Real age_in_years = age_in_seconds / seconds_per_year;
		// age_in_years = std::max(age_in_years, 1.0e-30); // age = 0 is allowed
		// Table coordinates: (age, mass) as specified in CSV input_names
		// std::array<amrex::Real, 2> const point = {age_in_years, mass_in_solar_masses};

		// Interpolate luminosity from table (returns array with nGroups elements)
		// Conversion from log space is handled automatically by DataTable::interpolate()
		// auto const luminosities = gpu_tables.luminosity.interpolate(point);

		// Update luminosity components (they are stored consecutively starting at lum_idx)
		if (burnState == Uninitialized) {
		  if (mass < MRADMIN) || (mdot == 0.0)) {
		  p,rdata(mlast_idx) = mass;
		    return;
		  }

		  // Initialize polytrope index
		  amrex::Real aGinit = 1.475 + 0.07*log10(mdot*YR_TO_SEC/C::M_solar);
		  amrex::Real npoly = 5.0 - 3.0/aGinit;
		  if (npoly < 1.5) npoly = 1.5;
		  if (npoly > 3.0) npoly = 3.0;

		  // Initialize radius
		  amrex::Real r = RSUN * max(2.5*pow(mdot*YR_TO_SEC/C::M_solar*1.0e5, 0.2), 2.0) 
		  burnState = None;
	       }

               // Update the radius 
               if (burnState != ZAMS) {
		 amrex::Real beta1 = beta(m);
		 amrex::Real dr = (2.0*mdot/m*r*(FK/(aG()*beta1)+1.0-1.0/(aG()*beta1))
				          + beta1/m * dlogBeta_dlogM(beta1) * mdot * r / beta1
			                   - 2.0/(beta1*aG())*r*r/(G*m*m)*(lStar()+eDotIon()-lDeut(beta1)));
		 amrex::Real rdottime = fabs(r/dr  )/100.0;
		 amrex::Real mdottime = fabs(m/mdot)/100.0;
    
		 if( rdottime < dt)
		   {
		     amrex::int rdotfac = ceil(dt/rdottime);
		     amrex::Real rdotfacr = rdotfac;
		     amrex::Real dtprime = dt/rdotfac;
		     // printf("In Loop: rdottime: %6.4e rdotfac: %i rdotfacr: %6.4e dtprime: %6.4e radius: %6.4e dt: %6.4e\n",rdottime,rdotfac,rdotfacr,dtprime,r,dt);
		     for(int rdotloop = 0; rdotloop < rdotfac; rdotloop++)
		       {
			 beta1 = beta(m);
			 dr = (2.0*mdot/m*r*(FK/(aG()*beta1)+1.0-1.0/(aG()*beta1))
			      + beta1/m * dlogBeta_dlogM(beta1) * mdot * r / beta1
			       - 2.0/(beta1*aG())*r*r/(G*m*m)*(lStar()+eDotIon()-lDeut(beta1)));
			 r += dtprime * dr;
		       }

		   } else if( mdottime < dt )
		   {
		     amrex::int mdotfac = ceil(dt/mdottime);
		     amrex::Real mdotfacr = mdotfac;
		     amrex::Real dtprime = dt/mdotfacr;
		     //printf("In Loop: mdottime: %6.4e mdotfac: %i mdotfacr: %6.4e dtprime: %6.4e mass: %6.4e mdot: %6.4e dt: %6.4e\n",mdottime,mdotfac,mdotfacr,dtprime,m,mdot,dt);
		     for(int mdotloop = 0; mdotloop < mdotfac; mdotloop++)
		       {
			 beta1=beta(m);
			 dr = (2.0*mdot/m*r*(FK/(aG()*beta1)+1.0-1.0/(aG()*beta1))
			       + beta1/m * dlogBeta_dlogM(beta1) * mdot * r / beta1
			       - 2.0/(beta1*aG())*r*r/(G*m*m)*(lStar()+eDotIon()-lDeut(beta1)));
			 r += dtprime * dr;
		       }
		   } else
		   {
		     beta1=beta(m);
		     dr = (2.0*mdot/m*r*(FK/(aG()*beta1)+1.0-1.0/(aG()*beta1))
			   + beta1/m * dlogBeta_dlogM(beta1) * mdot * r / beta1
			   - 2.0/(beta1*aG())*r*r/(G*m*m)*(lStar()+eDotIon()-lDeut(beta1)));
		     r += dt * dr;
		   }

		 if(r < 0.0e0)
		   {
		     r = 0.2*6.96e10; //Worst case and we do get a neg radius. reset it
		     pout() << "Star Particle updating radius: Found negative radius. Resetting to 0.2 R_sun" << std::endl;
		   }
	       }

               // Update the burning state and things associated with it
               switch (burnState) {

	       case None: {
		 // No burning yet, so check for the onset of D burning in the core
		 n = nInit(mdot);
		 if (Tc(m) > TDEUT) {
		   burnState = VariableCoreDeuterium;
		   n = 1.5; // Star becomes convective
		 }
		 break;
	       }

	       case VariableCoreDeuterium: {
		 // We are burning deuterium at a variable rate to keep the core
		 // temperature constant. Check to make sure we haven't exhausted
		 // our supply of D, in which case we change to steady core burning.
		 mdeut -= lDeut()*dt/PSID;
		 if ( mdeut <= mdot*dt ) {
		   burnState = SteadyCoreDeuterium;
		   mdeut = 0.0;
		 }
		 break;
	       }

	       case SteadyCoreDeuterium: {
		 // We are burning deuterium in the core at the rate it comes in. Check
		 // to see if a radiative barrier forms, which stops convection, shuts
		 // off core deuterium burning, and starts shell burning.
		 mdeut = 0.0;
		 if ( lDeut() <= FRAD*lZAMS() ) {
		   burnState = ShellDeuterium;
		   n = 3.0;
		   r *= SHELLFAC;
		 }
		 break;
	       }

	       case ShellDeuterium: {
		 // We are burning deuterium in a shell. Check if the radius has
		 // decreased to the ZAMS radius, in which case we stay on the ZAMS
		 // from now on.
		 mdeut = 0.0;
		 if ( r <= rZAMS() ) {
		   burnState = ZAMS;
		   r = rZAMS();
		 }
		 break;
	       }

	       case ZAMS: {
		 mdeut = 0.0;
		 break;
	       }
  }

  p.rdata(mlast_idx) = m;
}



  
		if (burnState == Uninitialized) {
		  p.rdata(lum_idx) = 0.0;
		}
		else {
		      // Determine burnState
		        // Do nothing if we are below the minimum mass or we have just been
  // created and thus don't have a valid mdot. Otherwise, if we're not
  // initilized, then initialize here
  if (burnState == Uninitialized) {
    if ((m < MRADMIN) || (mdot == 0.0)) {
      mlast = m;
      return;
    }
    n = nInit(mdot);
    r = radInit(mdot);
    burnState = None;

		  amrex::Real r = RSUN * max(2.5*pow(mdot*YR_TO_SEC/MSUN*1.0e5, 0.2), 2.0) );
		  amrex::Real lAcc = FACC * FK * G * mass * mdot / r;
		  amrex::Real ldisk = (1.0 - FK) * G * mass * mdot / r;
		  amrex::Real lsol = (ALPHA*pow(msol,5.5) + BETA*pow(msol,11)) / (GAMMA+pow(msol,3)+DELTA*pow(msol,5)+EPSILON*pow(msol,7)+ ZETA*pow(msol,8)+ETA*pow(msol,9.5));
		  lZAMS = lsol * LSUN;
		  lstar = IZAMS+lAcc;
		  amrex::Real Teff = pow(lstar / (4. * PI * r*r * SIGMA), 0.25);
                  if (Teff <= THAY) lstar = 4.*PI*r*r*SIGMA*pow(THAY, 4) );
		  p.rdata(lum_idx) =  lstar + ldisk;
		}
		
	}
};

#endif // AMREX_SPACEDIM == 3

} // namespace quokka

#endif // PARTICLE_RADIATION_HPP_
