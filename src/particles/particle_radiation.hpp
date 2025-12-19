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
	        const amrex::Real FACC = 0.5; // Fraction of accreted energy that comes out as radiation, rather than being advected into the stellar interior or used to drive a wind
		const amrex::Real FK = 0.5;     // Fraction of energy the falls into the sink particle but is radiated away from the the inner disk before reaching the stellar surface
	        const amrex::Real FWIND = 0.21; // Fraction of accreted mass ejected in a wind
		const amrex::Real FRAD = 0.33;  // A radiative barrier forms when L_deuterium <= FRAD * L_ZAMS. See McKee & Tan 2002
		const amrex::Real SHELLFAC = 2.1;  // Radius increases by SHELLFAC when shell burning starts
	        const amrex::Real THAY = 3000.0;     // Hayashi temperature
		const amrex::Real TDEUT = 1.5e6;  // Temperature when deuterium burning starts
		const amrex::Real PSIION = (16.0*C::ev2erg*C::n_A);  // Energy per gram needed to dissociate and ionize a molecular gas with solar abundances
		const amrex::Real PSID = (100*C::ev2erg*C::n_A);  // Energy per gram released by burning the deuterium in a gas with solar abundances
		const amrex::Real MRADMIN = (0.01*C::M_solar);  // Minimum mass at which we use the model
		const amrex::Real PI = std::acos(-1.0);
		  
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
		// Initialize polytrope index
		inline amrex::Real nInit(amrexReal::mdotInit) {
		  amrex::Real aGinit = 1.475 + 0.07*log10(mdotInit*seconds_per_year/C::M_solar);
		  amrex::Real npoly = 5.0 - 3.0/aGinit;
		  if (npoly < 1.5) npoly = 1.5;
		  if (npoly > 3.0) npoly = 3.0;
		  return( npoly );
		}
		
		// Initialize radius
		inline amrex::Real radInit(amrex::Real mdotInit) {
		  return (C::R_solar * max(2.5*pow(mdot*seconds_per_year/C::M_solar*1.0e5, 0.2), 2.0) ;
		}

	        // For a polytrope, the gravitational energy is aG G M^2 / R, aG = -3/(5-n)
		inline amrex::Real aG() {
		    return( 3.0/(5.0-n) );
		  }

		inline amrex::Real rhoc(amrex::Real mass) {
		  // Table of values of rho_mean / rho_c for n=1.5 to 3.1 in intervals of 0.1
		  static amrex::Real rhofactab[] = {
		       0.166931, 0.14742, 0.129933, 0.114265, 0.100242,
		       0.0877, 0.0764968, 0.0665109, 0.0576198, 0.0497216,
		       0.0427224, 0.0365357, 0.0310837, 0.0262952, 0.0221057,
		       0.0184553, 0.01529
		  };
		  amrex::int itab = (int) floor((n-1.5)/0.1);
		  amrex::Real wgt = (n - (1.5 + 0.1*itab)) / 0.1;
		  amrex::Real rhofac = rhofactab[itab]*(1.0-wgt) + rhofactab[itab+1]*wgt;
		  return( mass / (4./3.*PI*r*r*r) / rhofac );
		}

		// The central pressure in a polytropic model, found by table lookup.
		// See Kippenhahn & Weigert.
		inline amrex::Real Pc(Real mass) {
		  static amrex::Real pfactab[] = {
		       0.770087, 0.889001, 1.02979, 1.19731, 1.39753,
		       1.63818, 1.92909, 2.2825, 2.71504, 3.24792, 3.90921,
		       4.73657, 5.78067, 7.11088, 8.82286, 11.0515, 13.9885
		  };
		  amrex::int itab = (amrex::int) floor((n-1.5)/0.1);
		  amrex::Real wgt = (n - (1.5 + 0.1*itab)) / 0.1;
		  amrex::Real pfac = pfactab[itab]*(1.0-wgt) + pfactab[itab+1]*wgt;
		  return( pfac * G * mass*mass/(r*r*r*r) );
		}

		// The central temperature in a protostar, found by using a bisection
		// method to solve Pc = rho_c k Tc / (mu mH) + 1/3 a Tc^4.
		amrex::Real Tc(amrex::Real mass, amrex::Real rhoc1, amrex::Real Pc1) {
		  if (rhoc1 == -1.0) rhoc1 = rhoc(mass);
		  if (Pc1 == -1.0) Pc1 = Pc(mass);
		  const amrex::int JMAX = 40
		  const amrex::Real TOL = 1.0e-7
		  amrex::Real Tgas, Trad;
		  amrex::int j;
		  amrex::Real dx, f, fmid, xmid, rtb, x1, x2;
		  amrex::char errstr[256];

		  x1 = 0.0;
		  Tgas = Pc1*MU*MH/(C::k_B*rhoc1);
		  Trad = pow(3*Pc1/A, 0.25);
		  x2 = (Trad > Tgas) ? 2*Trad : 2*Tgas;
		  f = Pc1 - rhoc1*C::k_B*x1/(C::mu*C::m_p) - A*pow(x1,4)/3.0;
		  fmid=Pc1 - rhoc1*C::k_B*x2/(C::mu*C::m_p) - A*pow(x2,4)/3.0;
		  rtb = f < 0.0 ? (dx=x2-x1,x1) : (dx=x1-x2,x2);
		  for (j=1;j<=JMAX;j++) {
		    xmid=rtb+(dx *= 0.5);
		    fmid = Pc1 - rhoc1*C::k_B*xmid/(C::mu*C::m_p) - A*pow(xmid,4)/3.0;
		    if (fmid <= 0.0) rtb=xmid;
		    if (fabs(dx) < TOL*fabs(xmid) || fmid == 0.0) return rtb;
		  }
		}
		
		inline amrex::Real betac(amrex::Real mass, amrex::Real rhoc1, amrex::Real Pc1, amrex::Real Tc1) {
		  if (rhoc1 == -1.0) rhoc1 = rhoc(mass);
		  if (Pc1 == -1.0) Pc1 = Pc(mass);
		  if (Tc1 == -1.0) Tc1 = Tc(mass, rhoc1, Pc1);
		  return( rhoc1*C::k_B*Tc1/(C::mu*C::m_p) / Pc1 );
		}

		amrex::Real beta(amrex::Real mass, amrex::Real rhoc1, amrex::Real Pc1) {
		  if (n==3.0) {
		    // In this case we solve the Eddington quartic,
		    // P_c^3 = (3/a) (k / (mu mH))^4 (1 - beta) / beta^4 rho_c^4
		    // for beta
		    const amrex::int JMAX = 40;
		    const amrex::Real BETAMIN = 1.0e-4;
		    const amrex::Real BETAMAX = 1.0;
		    const amrex::Real TOL = 1.0e-7;
		    amrex::int j;
		    amrex::Real dx, f, fmid, xmid, rtb, x1, x2, coef;

		    if (rhoc1 == -1.0) rhoc1 = rhoc(mass);
		    if (Pc1 == -1.0) Pc1 = Pc(mass);
		    coef = 3/A*pow(C::k_B*rhoc1/(C::mu*C::m_p),4);
		    x1=BETAMIN;
		    x2=BETAMAX;
		    f = pow(Pc1,3) - coef * (1.0-x1)/pow(x1,4);
		    fmid = pow(Pc1,3) - coef * (1.0-x2)/pow(x2,4);
		    rtb = f < 0.0 ? (dx=x2-x1,x1) : (dx=x1-x2,x2);
		    for (j=1;j<=JMAX;j++) {
		      xmid=rtb+(dx *= 0.5);
		      fmid = pow(Pc1,3) - coef * (1.0-xmid)/pow(xmid,4);
		      if (fmid <= 0.0) rtb=xmid;
		      if (fabs(dx) < TOL*fabs(xmid) || fmid == 0.0) return rtb;
		    }
		    MayDay::
		      Error("SinkParticleData::beta(): bisection solve failed to converge");
		    return(-1);
		  } else {
		    // For npoly != 3, we use a table lookup. The values of beta have been
		    // pre-computed with mathematica. The table goes from M=5 to 50 solar
		    // masses in steps of 2.5 M_sun, and from n=1.5 to n=3 in steps of 0.5.
		    // We should never call this routine with M > 50 Msun, since by then
		    // the star should be fully on the main sequence.
		    const amrex::Real MTABMIN = (5.0*C::M_solar);
		    const amrex::Real MTABMAX = (50.0*C::M_solar);
		    const amrex::Real MTABSTEP = (2.5*C::M_solar);
		    const amrex::Real NTABMIN = 1.5;
		    const amrex::Real NTABMAX = 3.0;
		    const amrex::Real NTABSTEP = 0.5;
		    if (mass < MTABMIN) return(1.0);  // Set beta = 1 for M < 5 Msun
		    if ((mass >= MTABMAX) || (npoly >= NTABMAX)) {
		      MayDay::
			Error("SinkParticleData::beta(): off interpolation table");
		      return(-1.0);
		    }
		    static Real betatab[19][4] = {
		       {0.98785, 0.988928, 0.98947, 0.989634}, 
		       {0.97438, 0.976428, 0.977462, 0.977774}, 
		       {0.957927, 0.960895, 0.962397, 0.962846}, 
		       {0.939787, 0.943497, 0.945369, 0.945922}, 
		       {0.92091, 0.925151, 0.927276, 0.927896}, 
		       {0.901932, 0.906512, 0.908785, 0.909436}, 
		       {0.883254, 0.888017, 0.890353, 0.891013}, 
		       {0.865111, 0.86994, 0.872277, 0.872927}, 
		       {0.847635, 0.852445, 0.854739, 0.855367}, 
		       {0.830886, 0.835619, 0.837842, 0.838441}, 
		       {0.814885, 0.8195, 0.821635, 0.822201}, 
		       {0.799625, 0.804095, 0.806133, 0.806664}, 
		       {0.785082, 0.789394, 0.791328, 0.791825}, 
		       {0.771226, 0.775371, 0.777202, 0.777665}, 
		       {0.758022, 0.761997, 0.763726, 0.764156}, 
		       {0.745433, 0.749238, 0.750869, 0.751268}, 
		       {0.733423, 0.73706, 0.738596, 0.738966}, 
		       {0.721954, 0.725429, 0.726874, 0.727216}, 
		       {0.710993, 0.714311, 0.715671, 0.715987}
		    };
		    
		    // Locate ourselves on the table and do a linear interpolation
		    amrex::int midx = (int) floor((mass-MTABMIN)/MTABSTEP);
		    amrex::Real mwgt = (mass-(MTABMIN+midx*MTABSTEP)) / MTABSTEP;
		    amrex::int nidx = (int) floor((npoly-NTABMIN)/NTABSTEP);
		    amrex::Real nwgt = (npoly-(NTABMIN+nidx*NTABSTEP)) / NTABSTEP;
		    return ( betatab[midx][nidx]*(1.0-mwgt)*(1.0-nwgt) +
			     betatab[midx+1][nidx]*mwgt*(1.0-nwgt) +
			     betatab[midx][nidx+1]*(1.0-mwgt)*nwgt +
			     betatab[midx+1][nidx+1]*mwgt*nwgt );
		  }
		}
		
		const amrex::Real DM = (0.01*mass);
		amrex::Real dlogBetaOverBetac_dlogM(amrex::Real beta_1) {
		  // If npoly==3, beta = beta_c independent of M, so return 0
		  if (npoly==3) return(0.0);

		  // Otherwise take a numerical derivative
		  amrex::Real beta1;
		  if (beta_1==-1.0) beta1 = beta(mass);
		  else beta1 = beta_1;
		  amrex::Real beta2 = beta(mmass+DM);
		  amrex::Real betac1 = betac(mass);
		  amrex::Real betac2 = betac(mass+DM);
		  return( mass/(beta1/betac1) * ((beta2/betac2) - (beta1/betac1)) / DM );
		}

		amrex::Real dlogBeta_dlogM(amrex::Real beta_1) {
		  // Take a numerical derivative
		  amrex::Real beta1;
		  if (beta_1==-1.0) beta1 = beta(mass);
		  else beta1 = beta_1;
		  amrex::Real beta2 = beta(mass+DM);
		  return( mass/beta1 * (beta2-beta1) / DM );
		}

		amrex::Real luminosity() {
		  if (burnState == Uninitialized) return(0.0);
		  amrex::Real lum =  lStar() + lDisk();
		  //#ifdef NHIST
		  //SSRO take the smaller of the new luminosity or 
		  //25% greater than the old luminosity.
		  //l_hist=(lum<1.25*l_hist)?lum:1.25*l_hist;
		  //#endif
		  return( lum );
		}

		amrex::Real lStar() {
		  amrex::Real lstar = lZAMS() + lAcc();
		  amrex::Real Teff = pow(lstar / (4. * PI * r*r * C::sigma_SB), 0.25);
		  if (Teff > THAY) return( lstar );
		  else return( 4.*PI*r*r*C::sigma_SB*pow(THAY, 4) );
		}

		inline amrex::Real lAcc() {
		  return( FACC * FK * C::Gconst * mass * mdot / r );
		}

		amrex::Real lDisk() {
		  return ( (1.0 - FK) * C::Gconst * mass * mdot / r );
		}

		amrex::Real lDeut(amrex::Real beta1) {
		  switch (burnState) {
		  case Uninitialized: return(0.0);
		  case None: return(0.0);
		  case VariableCoreDeuterium: {
		    if (beta1 == -1.0) beta1=beta(mass);
		    return( lStar() + eDotIon() + C::Gconst*mass*mdot/r *
			    (1.0 - FK - aG()*beta1/2.0 *
			    (1.0 + dlogBetaOverBetac_dlogM(beta1))) );
		  }
		  case SteadyCoreDeuterium: return( mdot * PSID );
		  case ShellDeuterium: return( mdot * PSID );
		  default: MayDay::Error("SinkParticleData::lDeut(): bad value of burnState");
		  }
		  return(-1.0); // Never get here
		}
  
		inline amrex::Real eDotIon() {
		  return( mdot * PSIION );
		}

		inline amrex::Realc dlogR_dlogM(amrex::Real beta1) {
		  if (beta1==-1.0) beta1 = beta(m);
		  return( 2.0 - 2.0/(aG()*beta1) * (1.0 - FK) +
			  dlogBeta_dlogM(beta1) - 2.0*r/(aG()*beta1*C::Gconst*mass*mdot) * 
			  (lStar() + eDotIon() - lDeut(beta1)) );
		}

		amrex::Real vWind(amrex::Real vw_fkep, amrex::Real vw_max, amrex::Real rlaunch) {
		  // Set wind velocity equal to Keplerian velocity
		  if (burnState == Uninitialized) return(0.0);
		  return( min(vw_fkep*sqrt(G*m/r),vw_max) * sqrt(1.+2.*r/(pow(vw_fkep,2)*rlaunch)));
		}


		
		if (burnState == Uninitialized) {
		  if (mass < MRADMIN) || (mdot == 0.0)) {
		  p,rdata(mlast_idx) = mass;
		    return;
		  }
		amrex::Real npoly = nInit(mdot);
		amrex::Real r = radInit(mdot);
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

    amrex::Real r = C::R_solar * max(2.5*pow(mdot*YR_TO_SEC/C::M_solar*1.0e5, 0.2), 2.0) );
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
