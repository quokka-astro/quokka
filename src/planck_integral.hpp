//==============================================================================
//==============================================================================
/// \file planck_integral.hpp
/// \brief Some functions for quickly integrating the Planck function.
///

#ifndef PLANCKINTEGRAL_HPP_ // NOLINT
#define PLANCKINTEGRAL_HPP_

#include <algorithm>
#include <cmath>

#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_BLassert.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_REAL.H"

#include "valarray.hpp"

using Real = amrex::Real;

static constexpr bool USE_SECOND_ORDER = false;
static constexpr double PI = M_PI;
static constexpr Real gInf = PI * PI * PI * PI / 15.0;
static constexpr int INTERP_SIZE = 1000;
static constexpr Real LOG_X_MIN = -3.;
static constexpr Real LOG_X_MAX = 2.;
static constexpr Real Y_INTERP_MIN = 5.13106651231913e-11;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto interpolate_planck_integral(Real logx) -> Real
{
	const amrex::GpuArray<Real, INTERP_SIZE> Y_interp = {
	    5.13106651231913e-11, 5.31154385234633e-11, 5.49836892699242e-11, 5.69176498520072e-11, 5.89196312693410e-11, 6.09920257923993e-11,
	    6.31373098202748e-11, 6.53580468388731e-11, 6.76568904831568e-11, 7.00365877070177e-11, 7.24999820646129e-11, 7.50500171070513e-11,
	    7.76897398985101e-11, 8.04223046559504e-11, 8.32509765168210e-11, 8.61791354391650e-11, 8.92102802389228e-11, 9.23480327690886e-11,
	    9.55961422458248e-11, 9.89584897266783e-11, 1.02439092746167e-10, 1.06042110114433e-10, 1.09771846884475e-10, 1.13632759494054e-10,
	    1.17629461088394e-10, 1.21766727029926e-10, 1.26049500601738e-10, 1.30482898911574e-10, 1.35072219003263e-10, 1.39822944183133e-10,
	    1.44740750568632e-10, 1.49831513867320e-10, 1.55101316394005e-10, 1.60556454334688e-10, 1.66203445265668e-10, 1.72049035937109e-10,
	    1.78100210330137e-10, 1.84364197997101e-10, 1.90848482695078e-10, 1.97560811322773e-10, 2.04509203171551e-10, 2.11701959501695e-10,
	    2.19147673455157e-10, 2.26855240316808e-10, 2.34833868136268e-10, 2.43093088723141e-10, 2.51642769028618e-10, 2.60493122927240e-10,
	    2.69654723412582e-10, 2.79138515221631e-10, 2.88955827902917e-10, 2.99118389343854e-10, 3.09638339773471e-10, 3.20528246257342e-10,
	    3.31801117701924e-10, 3.43470420386046e-10, 3.55550094038453e-10, 3.68054568480216e-10, 3.80998780851978e-10, 3.94398193446712e-10,
	    4.08268812169028e-10, 4.22627205643315e-10, 4.37490524993112e-10, 4.52876524315756e-10, 4.68803581876506e-10, 4.85290722047184e-10,
	    5.02357638015996e-10, 5.20024715295102e-10, 5.38313056054391e-10, 5.57244504310193e-10, 5.76841671998992e-10, 5.97127965967462e-10,
	    6.18127615910574e-10, 6.39865703291578e-10, 6.62368191277859e-10, 6.85661955728858e-10, 7.09774817272447e-10, 7.34735574508390e-10,
	    7.60574038378395e-10, 7.87321067743615e-10, 8.15008606211976e-10, 8.43669720259491e-10, 8.73338638690733e-10, 9.04050793485473e-10,
	    9.35842862080331e-10, 9.68752811135927e-10, 1.00281994184110e-09, 1.03808493680913e-09, 1.07458990862075e-09, 1.11237845007285e-09,
	    1.15149568619174e-09, 1.19198832807383e-09, 1.23390472861740e-09, 1.27729494021164e-09, 1.32221077445233e-09, 1.36870586395480e-09,
	    1.41683572633772e-09, 1.46665783045459e-09, 1.51823166495085e-09, 1.57161880922941e-09, 1.62688300690764e-09, 1.68409024185502e-09,
	    1.74330881690064e-09, 1.80460943530470e-09, 1.86806528509144e-09, 1.93375212634329e-09, 2.00174838156058e-09, 2.07213522919407e-09,
	    2.14499670046193e-09, 2.22041977956665e-09, 2.29849450743035e-09, 2.37931408907255e-09, 2.46297500475883e-09, 2.54957712505093e-09,
	    2.63922382989713e-09, 2.73202213190352e-09, 2.82808280393268e-09, 2.92752051118301e-09, 3.03045394790399e-09, 3.13700597891162e-09,
	    3.24730378607118e-09, 3.36147901992283e-09, 3.47966795662940e-09, 3.60201166043326e-09, 3.72865615181626e-09, 3.85975258156164e-09,
	    3.99545741092598e-09, 4.13593259813458e-09, 4.28134579142278e-09, 4.43187052885253e-09, 4.58768644514093e-09, 4.74897948574911e-09,
	    4.91594212848284e-09, 5.08877361287036e-09, 5.26768017758986e-09, 5.45287530622875e-09, 5.64457998166574e-09, 5.84302294938023e-09,
	    6.04844098999986e-09, 6.26107920141155e-09, 6.48119129077019e-09, 6.70903987675356e-09, 6.94489680242001e-09, 7.18904345904333e-09,
	    7.44177112130790e-09, 7.70338129426295e-09, 7.97418607244730e-09, 8.25450851161225e-09, 8.54468301348335e-09, 8.84505572401747e-09,
	    9.15598494562977e-09, 9.47784156387837e-09, 9.81100948911386e-09, 1.01558861136183e-08, 1.05128827847758e-08, 1.08824252948373e-08,
	    1.12649543878593e-08, 1.16609262844191e-08, 1.20708132247305e-08, 1.24951040308010e-08, 1.29343046883010e-08, 1.33889389488332e-08,
	    1.38595489533183e-08, 1.43466958772336e-08, 1.48509605984739e-08, 1.53729443886220e-08, 1.59132696284536e-08, 1.64725805485203e-08,
	    1.70515439956924e-08, 1.76508502265666e-08, 1.82712137286841e-08, 1.89133740705273e-08, 1.95780967813046e-08, 2.02661742615677e-08,
	    2.09784267257370e-08, 2.17157031776542e-08, 2.24788824203168e-08, 2.32688741009903e-08, 2.40866197929379e-08, 2.49330941150455e-08,
	    2.58093058906704e-08, 2.67162993470852e-08, 2.76551553569361e-08, 2.86269927231853e-08, 2.96329695090607e-08, 3.06742844145833e-08,
	    3.17521782013019e-08, 3.28679351669244e-08, 3.40228846715862e-08, 3.52184027175616e-08, 3.64559135842889e-08, 3.77368915206421e-08,
	    3.90628624964467e-08, 4.04354060153165e-08, 4.18561569909489e-08, 4.33268076891010e-08, 4.48491097375368e-08, 4.64248762063262e-08,
	    4.80559837609480e-08, 4.97443748907457e-08, 5.14920602153636e-08, 5.33011208718921e-08, 5.51737109855357e-08, 5.71120602267269e-08,
	    5.91184764576961e-08, 6.11953484716345e-08, 6.33451488276664e-08, 6.55704367849915e-08, 6.78738613396475e-08, 7.02581643674821e-08,
	    7.27261838770387e-08, 7.52808573761910e-08, 7.79252253564979e-08, 8.06624348993883e-08, 8.34957434084167e-08, 8.64285224720030e-08,
	    8.94642618611955e-08, 9.26065736671698e-08, 9.58591965833425e-08, 9.92260003371381e-08, 1.02710990276622e-07, 1.06318312117416e-07,
	    1.10052256855463e-07, 1.13917265851439e-07, 1.17917936092783e-07, 1.22059025639546e-07, 1.26345459260453e-07, 1.30782334265818e-07,
	    1.35374926544155e-07, 1.40128696809597e-07, 1.45049297067452e-07, 1.50142577305509e-07, 1.55414592418927e-07, 1.60871609376877e-07,
	    1.66520114639321e-07, 1.72366821832661e-07, 1.78418679693242e-07, 1.84682880288050e-07, 1.91166867522261e-07, 1.97878345943570e-07,
	    2.04825289853699e-07, 2.12015952737698e-07, 2.19458877022141e-07, 2.27162904173630e-07, 2.35137185149444e-07, 2.43391191212569e-07,
	    2.51934725123801e-07, 2.60777932723985e-07, 2.69931314919986e-07, 2.79405740088385e-07, 2.89212456911453e-07, 2.99363107660370e-07,
	    3.09869741941260e-07, 3.20744830920096e-07, 3.32001282043091e-07, 3.43652454269803e-07, 3.55712173836716e-07, 3.68194750569735e-07,
	    3.81114994764608e-07, 3.94488234655014e-07, 4.08330334488670e-07, 4.22657713232549e-07, 4.37487363929037e-07, 4.52836873725572e-07,
	    4.68724444601132e-07, 4.85168914813694e-07, 5.02189781093672e-07, 5.19807221609161e-07, 5.38042119729712e-07, 5.56916088616320e-07,
	    5.76451496666181e-07, 5.96671493841865e-07, 6.17600038915484e-07, 6.39261927659489e-07, 6.61682822016903e-07, 6.84889280284839e-07,
	    7.08908788346327e-07, 7.33769791986730e-07, 7.59501730332203e-07, 7.86135070449025e-07, 8.13701343143856e-07, 8.42233180006450e-07,
	    8.71764351737712e-07, 9.02329807807497e-07, 9.33965717488061e-07, 9.66709512310597e-07, 1.00059992999406e-06, 1.03567705989696e-06,
	    1.07198239004473e-06, 1.10955885578699e-06, 1.14845089014086e-06, 1.18870447587844e-06, 1.23036719941860e-06, 1.27348830658524e-06,
	    1.31811876029618e-06, 1.36431130024932e-06, 1.41212050467473e-06, 1.46160285422377e-06, 1.51281679806876e-06, 1.56582282228930e-06,
	    1.62068352062374e-06, 1.67746366766725e-06, 1.73623029460044e-06, 1.79705276753560e-06, 1.86000286857032e-06, 1.92515487964158e-06,
	    1.99258566927638e-06, 2.06237478233835e-06, 2.13460453287302e-06, 2.20936010015820e-06, 2.28672962806914e-06, 2.36680432787228e-06,
	    2.44967858456492e-06, 2.53545006688239e-06, 2.62421984109816e-06, 2.71609248874684e-06, 2.81117622840426e-06, 2.90958304166344e-06,
	    3.01142880344982e-06, 3.11683341682425e-06, 3.22592095242699e-06, 3.33881979272121e-06, 3.45566278120004e-06, 3.57658737672636e-06,
	    3.70173581318078e-06, 3.83125526459847e-06, 3.96529801598244e-06, 4.10402163998630e-06, 4.24758917966682e-06, 4.39616933751279e-06,
	    4.54993667096393e-06, 4.70907179464063e-06, 4.87376158951296e-06, 5.04419941924458e-06, 5.22058535395572e-06, 5.40312640165709e-06,
	    5.59203674761520e-06, 5.78753800191829e-06, 5.98985945552132e-06, 6.19923834505712e-06, 6.41592012671129e-06, 6.64015875946761e-06,
	    6.87221699804145e-06, 7.11236669582908e-06, 7.36088911821165e-06, 7.61807526656412e-06, 7.88422621333094e-06, 8.15965344854206e-06,
	    8.44467923815610e-06, 8.73963699462922e-06, 9.04487166012244e-06, 9.36074010277339e-06, 9.68761152647257e-06, 1.00258678945990e-05,
	    1.03759043681853e-05, 1.07381297589965e-05, 1.11129669980264e-05, 1.15008536199259e-05, 1.19022422639019e-05, 1.23176011916376e-05,
	    1.27474148228044e-05, 1.31921842887576e-05, 1.36524280050226e-05, 1.41286822632020e-05, 1.46215018429526e-05, 1.51314606447043e-05,
	    1.56591523438130e-05, 1.62051910668640e-05, 1.67702120908643e-05, 1.73548725660874e-05, 1.79598522633599e-05, 1.85858543466014e-05,
	    1.92336061714604e-05, 1.99038601109128e-05, 2.05973944087194e-05, 2.13150140616668e-05, 2.20575517315478e-05, 2.28258686878661e-05,
	    2.36208557822846e-05, 2.44434344558664e-05, 2.52945577801947e-05, 2.61752115334906e-05, 2.70864153128846e-05, 2.80292236840348e-05,
	    2.90047273693233e-05, 3.00140544759009e-05, 3.10583717648929e-05, 3.21388859631176e-05, 3.32568451187162e-05, 3.44135400021342e-05,
	    3.56103055539415e-05, 3.68485223810271e-05, 3.81296183027500e-05, 3.94550699486816e-05, 4.08264044096238e-05, 4.22452009436425e-05,
	    4.37130927389094e-05, 4.52317687352034e-05, 4.68029755059793e-05, 4.84285192029731e-05, 5.01102675653738e-05, 5.18501519956561e-05,
	    5.36501697042345e-05, 5.55123859251645e-05, 5.74389362051880e-05, 5.94320287684922e-05, 6.14939469596215e-05, 6.36270517670624e-05,
	    6.58337844300949e-05, 6.81166691315870e-05, 7.04783157794903e-05, 7.29214228798795e-05, 7.54487805044671e-05, 7.80632733556127e-05,
	    8.07678839319405e-05, 8.35656957977715e-05, 8.64598969596782e-05, 8.94537833535650e-05, 9.25507624457860e-05, 9.57543569519146e-05,
	    9.90682086768900e-05, 1.02496082480377e-04, 1.06041870371293e-04, 1.09709595735569e-04, 1.13503417701344e-04, 1.17427635645901e-04,
	    1.21486693848798e-04, 1.25685186295771e-04, 1.30027861638117e-04, 1.34519628312413e-04, 1.39165559825568e-04, 1.43970900210344e-04,
	    1.48941069656646e-04, 1.54081670324015e-04, 1.59398492340942e-04, 1.64897519996760e-04, 1.70584938132055e-04, 1.76467138733688e-04,
	    1.82550727740727e-04, 1.88842532067727e-04, 1.95349606852017e-04, 2.02079242931813e-04, 2.09038974562194e-04, 2.16236587376153e-04,
	    2.23680126598151e-04, 2.31377905517818e-04, 2.39338514231622e-04, 2.47570828660597e-04, 2.56084019852395e-04, 2.64887563576185e-04,
	    2.73991250219142e-04, 2.83405194993507e-04, 2.93139848463459e-04, 3.03206007401258e-04, 3.13614825982402e-04, 3.24377827329785e-04,
	    3.35506915417118e-04, 3.47014387342128e-04, 3.58912945980356e-04, 3.71215713030623e-04, 3.83936242463556e-04, 3.97088534384830e-04,
	    4.10687049325092e-04, 4.24746722968853e-04, 4.39282981334918e-04, 4.54311756421260e-04, 4.69849502327567e-04, 4.85913211868988e-04,
	    5.02520433695000e-04, 5.19689289927584e-04, 5.37438494333298e-04, 5.55787371044156e-04, 5.74755873842591e-04, 5.94364606026140e-04,
	    6.14634840867838e-04, 6.35588542688710e-04, 6.57248388559093e-04, 6.79637790645931e-04, 7.02780919223541e-04, 7.26702726365773e-04,
	    7.51428970337835e-04, 7.76986240706518e-04, 8.03401984187863e-04, 8.30704531251831e-04, 8.58923123503838e-04, 8.88087941863510e-04,
	    9.18230135561360e-04, 9.49381851974567e-04, 9.81576267323403e-04, 1.01484761825029e-03, 1.04923123430391e-03, 1.08476357135114e-03,
	    1.12148224594014e-03, 1.15942607063813e-03, 1.19863509036803e-03, 1.23915061976838e-03, 1.28101528160148e-03, 1.32427304623501e-03,
	    1.36896927222285e-03, 1.41515074801118e-03, 1.46286573479639e-03, 1.51216401056158e-03, 1.56309691531896e-03, 1.61571739758577e-03,
	    1.67008006212152e-03, 1.72624121895508e-03, 1.78425893373000e-03, 1.84419307939718e-03, 1.90610538928399e-03, 1.97005951156939e-03,
	    2.03612106519481e-03, 2.10435769724076e-03, 2.17483914179945e-03, 2.24763728037369e-03, 2.32282620383274e-03, 2.40048227595577e-03,
	    2.48068419859368e-03, 2.56351307848021e-03, 2.64905249572309e-03, 2.73738857400637e-03, 2.82861005253444e-03, 2.92280835974870e-03,
	    3.02007768884749e-03, 3.12051507513950e-03, 3.22422047526104e-03, 3.33129684828692e-03, 3.44185023876447e-03, 3.55598986169986e-03,
	    3.67382818952517e-03, 3.79548104107444e-03, 3.92106767259592e-03, 4.05071087082737e-03, 4.18453704816015e-03, 4.32267633991725e-03,
	    4.46526270376913e-03, 4.61243402131047e-03, 4.76433220181944e-03, 4.92110328822008e-03, 5.08289756526679e-03, 5.24986966996850e-03,
	    5.42217870426840e-03, 5.59998834999334e-03, 5.78346698608510e-03, 5.97278780812371e-03, 6.16812895015058e-03, 6.36967360879717e-03,
	    6.57761016972213e-03, 6.79213233635708e-03, 7.01343926095870e-03, 7.24173567796120e-03, 7.47723203962037e-03, 7.72014465393671e-03,
	    7.97069582484165e-03, 8.22911399462660e-03, 8.49563388859102e-03, 8.77049666188060e-03, 9.05395004848263e-03, 9.34624851234054e-03,
	    9.64765340054414e-03, 9.95843309854735e-03, 1.02788631873588e-02, 1.06092266026451e-02, 1.09498137956801e-02, 1.13009228960672e-02,
	    1.16628598761536e-02, 1.20359387170507e-02, 1.24204815761636e-02, 1.28168189561279e-02, 1.32252898750421e-02, 1.36462420378753e-02,
	    1.40800320089211e-02, 1.45270253851585e-02, 1.49875969703717e-02, 1.54621309498700e-02, 1.59510210656365e-02, 1.64546707917266e-02,
	    1.69734935097231e-02, 1.75079126840422e-02, 1.80583620368742e-02, 1.86252857225298e-02, 1.92091385009450e-02, 1.98103859100909e-02,
	    2.04295044370138e-02, 2.10669816872190e-02, 2.17233165520966e-02, 2.23990193740705e-02, 2.30946121091360e-02, 2.38106284864334e-02,
	    2.45476141644905e-02, 2.53061268837446e-02, 2.60867366149404e-02, 2.68900257029786e-02, 2.77165890057714e-02, 2.85670340276416e-02,
	    2.94419810467820e-02, 3.03420632362686e-02, 3.12679267781039e-02, 3.22202309697415e-02, 3.31996483225226e-02, 3.42068646514326e-02,
	    3.52425791555635e-02, 3.63075044886435e-02, 3.74023668189739e-02, 3.85279058780866e-02, 3.96848749974164e-02, 4.08740411322525e-02,
	    4.20961848722151e-02, 4.33521004374744e-02, 4.46425956599076e-02, 4.59684919483651e-02, 4.73306242371909e-02, 4.87298409171219e-02,
	    5.01670037476641e-02, 5.16429877500212e-02, 5.31586810796296e-02, 5.47149848773299e-02, 5.63128130981828e-02, 5.79530923169195e-02,
	    5.96367615089918e-02, 6.13647718061721e-02, 6.31380862256328e-02, 6.49576793714191e-02, 6.68245371072120e-02, 6.87396561992674e-02,
	    7.07040439284000e-02, 7.27187176698751e-02, 7.47847044400598e-02, 7.69030404086791e-02, 7.90747703755196e-02, 8.13009472104207e-02,
	    8.35826312553950e-02, 8.59208896877236e-02, 8.83167958428806e-02, 9.07714284961499e-02, 9.32858711018127e-02, 9.58612109888042e-02,
	    9.84985385117569e-02, 1.01198946156378e-01, 1.03963527598137e-01, 1.06793376713280e-01, 1.09689586541221e-01, 1.12653248197418e-01,
	    1.15685449735882e-01, 1.18787274960537e-01, 1.21959802184706e-01, 1.25204102938068e-01, 1.28521240620515e-01, 1.31912269102421e-01,
	    1.35378231270922e-01, 1.38920157521908e-01, 1.42539064197535e-01, 1.46235951969189e-01, 1.50011804165931e-01, 1.53867585048623e-01,
	    1.57804238030030e-01, 1.61822683841385e-01, 1.65923818646018e-01, 1.70108512100866e-01, 1.74377605366804e-01, 1.78731909068959e-01,
	    1.83172201208350e-01, 1.87699225026393e-01, 1.92313686824032e-01, 1.97016253737475e-01, 2.01807551472730e-01, 2.06688162001400e-01,
	    2.11658621220406e-01, 2.16719416578607e-01, 2.21870984673491e-01, 2.27113708821437e-01, 2.32447916605271e-01, 2.37873877403165e-01,
	    2.43391799903184e-01, 2.49001829608086e-01, 2.54704046335277e-01, 2.60498461717130e-01, 2.66385016707149e-01, 2.72363579097795e-01,
	    2.78433941056065e-01, 2.84595816683222e-01, 2.90848839605379e-01, 2.97192560601924e-01, 3.03626445279045e-01, 3.10149871795929e-01,
	    3.16762128651441e-01, 3.23462412539369e-01, 3.30249826280558e-01, 3.37123376840500e-01, 3.44081973441132e-01, 3.51124425775831e-01,
	    3.58249442336735e-01, 3.65455628863676e-01, 3.72741486924167e-01, 3.80105412633931e-01, 3.87545695527574e-01, 3.95060517589021e-01,
	    4.02647952451323e-01, 4.10305964775436e-01, 4.18032409817471e-01, 4.25825033193802e-01, 4.33681470853283e-01, 4.41599249265575e-01,
	    4.49575785834375e-01, 4.57608389544002e-01, 4.65694261847460e-01, 4.73830497803686e-01, 4.82014087471223e-01, 4.90241917565055e-01,
	    4.98510773382759e-01, 5.06817341005508e-01, 5.15158209778763e-01, 5.23529875076768e-01, 5.31928741354150e-01, 5.40351125487079e-01,
	    5.48793260405529e-01, 5.57251299017233e-01, 5.65721318422905e-01, 5.74199324421230e-01, 5.82681256301048e-01, 5.91162991916975e-01,
	    5.99640353043535e-01, 6.08109111001650e-01, 6.16564992550088e-01, 6.25003686033168e-01, 6.33420847774753e-01, 6.41812108707236e-01,
	    6.50173081222893e-01, 6.58499366233683e-01, 6.66786560424242e-01, 6.75030263681525e-01, 6.83226086683283e-01, 6.91369658626298e-01,
	    6.99456635074111e-01, 7.07482705902802e-01, 7.15443603322284e-01, 7.23335109949547e-01, 7.31153066909284e-01, 7.38893381936499e-01,
	    7.46552037454865e-01, 7.54125098603921e-01, 7.61608721187616e-01, 7.68999159516237e-01, 7.76292774113401e-01, 7.83486039259588e-01,
	    7.90575550343602e-01, 7.97558030993412e-01, 8.04430339958031e-01, 8.11189477712446e-01, 8.17832592758135e-01, 8.24356987592360e-01,
	    8.30760124320269e-01, 8.37039629884807e-01, 8.43193300890589e-01, 8.49219107999183e-01, 8.55115199874696e-01, 8.60879906660147e-01,
	    8.66511742966859e-01, 8.72009410360971e-01, 8.77371799333161e-01, 8.82597990739802e-01, 8.87687256705985e-01, 8.92639060983172e-01,
	    8.97453058756640e-01, 9.02129095900354e-01, 9.06667207679436e-01, 9.11067616902956e-01, 9.15330731532388e-01, 9.19457141753639e-01,
	    9.23447616523196e-01, 9.27303099601457e-01, 9.31024705088866e-01, 9.34613712482908e-01, 9.38071561276434e-01, 9.41399845120012e-01,
	    9.44600305573244e-01, 9.47674825471970e-01, 9.50625421940204e-01, 9.53454239077385e-01, 9.56163540353069e-01, 9.58755700742597e-01,
	    9.61233198638420e-01, 9.63598607572757e-01, 9.65854587788012e-01, 9.68003877691919e-01, 9.70049285234675e-01, 9.71993679245431e-01,
	    9.73839980765334e-01, 9.75591154413937e-01, 9.77250199825205e-01, 9.78820143188481e-01, 9.80304028928764e-01, 9.81704911559357e-01,
	    9.83025847738525e-01, 9.84269888560126e-01, 9.85440072106400e-01, 9.86539416289091e-01, 9.87570912002989e-01, 9.88537516613726e-01,
	    9.89442147799308e-01, 9.90287677762442e-01, 9.91076927828214e-01, 9.91812663439099e-01, 9.92497589556751e-01, 9.93134346477390e-01,
	    9.93725506065078e-01, 9.94273568404610e-01, 9.94780958873287e-01, 9.95250025628410e-01, 9.95683037505046e-01, 9.96082182316375e-01,
	    9.96449565546870e-01, 9.96787209426587e-01, 9.97097052373068e-01, 9.97380948785679e-01, 9.97640669175781e-01, 9.97877900614793e-01,
	    9.98094247481114e-01, 9.98291232485922e-01, 9.98470297957150e-01, 9.98632807360337e-01, 9.98780047034705e-01, 9.98913228122613e-01,
	    9.99033488670495e-01, 9.99141895879542e-01, 9.99239448484695e-01, 9.99327079240949e-01, 9.99405657496550e-01, 9.99475991833401e-01,
	    9.99538832755780e-01, 9.99594875409427e-01, 9.99644762314034e-01, 9.99689086093280e-01, 9.99728392187655e-01, 9.99763181536544e-01,
	    9.99793913217186e-01, 9.99821007029420e-01, 9.99844846016313e-01, 9.99865778911996e-01, 9.99884122509270e-01, 9.99900163940662e-01,
	    9.99914162867806e-01, 9.99926353575078e-01, 9.99936946964462e-01, 9.99946132449588e-01, 9.99954079747820e-01, 9.99960940570086e-01,
	    9.99966850208925e-01, 9.99971929025915e-01, 9.99976283840292e-01, 9.99980009221072e-01, 9.99983188685505e-01, 9.99985895807088e-01,
	    9.99988195236654e-01, 9.99990143640393e-01, 9.99991790558803e-01, 9.99993179190753e-01, 9.99994347106922e-01, 9.99995326896920e-01,
	    9.99996146754400e-01, 9.99996831004433e-01, 9.99997400577326e-01, 9.99997873432983e-01, 9.99998264939748e-01, 9.99998588211549e-01,
	    9.99998854406949e-01, 9.99999072993575e-01, 9.99999251981168e-01, 9.99999398126314e-01, 9.99999517111706e-01, 9.99999613702597e-01,
	    9.99999691882891e-01, 9.99999754973131e-01, 9.99999805732451e-01, 9.99999846446375e-01, 9.99999879002170e-01, 9.99999904953301e-01,
	    9.99999925574369e-01, 9.99999941907787e-01, 9.99999954803277e-01, 9.99999964951203e-01, 9.99999972910575e-01, 9.99999979132518e-01,
	    9.99999983979839e-01, 9.99999987743315e-01, 9.99999990655177e-01, 9.99999992900246e-01, 9.99999994625081e-01, 9.99999995945485e-01,
	    9.99999996952625e-01, 9.99999997718006e-01, 9.99999998297504e-01, 9.99999998734618e-01, 9.99999999063081e-01, 9.99999999308952e-01,
	    9.99999999492285e-01, 9.99999999628449e-01, 9.99999999729179e-01, 9.99999999803396e-01, 9.99999999857856e-01, 9.99999999897654e-01,
	    9.99999999926617e-01, 9.99999999947606e-01, 9.99999999962751e-01, 9.99999999973633e-01, 9.99999999981418e-01, 9.99999999986962e-01,
	    9.99999999990893e-01, 9.99999999993667e-01, 9.99999999995617e-01, 9.99999999996980e-01, 9.99999999997929e-01, 9.99999999998587e-01,
	    9.99999999999040e-01, 9.99999999999351e-01, 9.99999999999564e-01, 9.99999999999708e-01, 9.99999999999806e-01, 9.99999999999871e-01,
	    9.99999999999915e-01, 9.99999999999945e-01, 9.99999999999964e-01, 9.99999999999977e-01, 9.99999999999985e-01, 9.99999999999990e-01,
	    9.99999999999994e-01, 9.99999999999996e-01, 9.99999999999998e-01, 9.99999999999999e-01, 9.99999999999999e-01, 9.99999999999999e-01,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00,
	    1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00, 1.00000000000000e+00};
	const int arr_len = INTERP_SIZE;
	const int j = static_cast<int>((logx - LOG_X_MIN) / (LOG_X_MAX - LOG_X_MIN) * (arr_len - 1));
	const Real gap = (LOG_X_MAX - LOG_X_MIN) / (arr_len - 1);

	Real y = NAN;
	if ((j < 0) || (j >= arr_len - 1)) {
		return y;
	}
	if constexpr (!USE_SECOND_ORDER) {
		// linear interpolation
		const Real slope = (Y_interp[j + 1] - Y_interp[j]) / gap;
		y = slope * (logx - (LOG_X_MIN + j * gap)) + Y_interp[j];
	} else {
		if (j >= arr_len - 2) {
			// linear interpolation
			const Real slope = (Y_interp[j + 1] - Y_interp[j]) / gap;
			y = slope * (logx - (LOG_X_MIN + j * gap)) + Y_interp[j];
		} else {
			// quadratic interpolation
			const Real slope = (Y_interp[j + 1] - Y_interp[j]) / gap;
			const Real slope2 = (Y_interp[j + 2] - Y_interp[j + 1]) / gap;
			const Real slope3 = (slope2 - slope) / (2.0 * gap);
			const Real x0 = LOG_X_MIN + j * gap;
			const Real x1 = LOG_X_MIN + (j + 1) * gap;
			y = Y_interp[j] + slope * (logx - x0) + slope3 * (logx - x0) * (logx - x1);
		}
	}
	return y;
}

// Integrate the Planck integral, x^3 / (exp(x) - 1), from 0 to x. Return its ratio to the integral from 0 to infinity (pi^4 / 15).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto integrate_planck_from_0_to_x(const Real x) -> Real
{
	AMREX_ASSERT(x >= 0.);

	if (x <= 0.) {
		return 0.;
	}

	const Real logx = std::log10(x);
	Real y = NAN;
	if (logx < LOG_X_MIN) {
		// y = x * x * x / 3.0;    // 1st order
		y = (-4 + x) * x + 8 * std::log((2 + x) / 2); // 2nd order
		// AMREX_ASSERT(y <= Y_INTERP_MIN);
    if (y > Y_INTERP_MIN) {
      y = Y_INTERP_MIN;
    } else if (y < 0.) {
      y = 0.;
    }
	} else if (logx >= LOG_X_MAX) {
		return 1.0;
	} else {
		y = interpolate_planck_integral(logx);
	}
	assert(!isnan(y));
	assert(y <= 1.);
	return y;
}

#endif // PLANCKINTEGRAL_HPP_