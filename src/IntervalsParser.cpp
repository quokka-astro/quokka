#include "IntervalsParser.H"

#include "AMReX_BLassert.H"
#include <AMReX_Utility.H>

#include <algorithm>
#include <memory>

using amrex::Parser;


double
parseStringtoReal(std::string str)
{
    auto parser = makeParser(str, {});
    auto exe = parser.compileHost<0>();
    double result = exe();
    return result;
}

void Store_parserString(const amrex::ParmParse& pp, std::string query_string,
                        std::string& stored_string)
{
    std::vector<std::string> f;
    pp.getarr(query_string.c_str(), f);
    stored_string.clear();
    for (auto const& s : f) {
        stored_string += s;
    }
    f.clear();
}

int
queryWithParser (const amrex::ParmParse& a_pp, char const * const str, double& val)
{
    // call amrex::ParmParse::query, check if the user specified str.
    std::string tmp_str;
    int is_specified = a_pp.query(str, tmp_str);
    if (is_specified)
    {
        // If so, create a parser object and apply it to the value provided by the user.
        std::string str_val;
        Store_parserString(a_pp, str, str_val);
        val = parseStringtoReal(str_val);
    }
    // return the same output as amrex::ParmParse::query
    return is_specified;
}

Parser makeParser (std::string const& parse_function, amrex::Vector<std::string> const& varnames)
{
    // Since queryWithParser recursively calls this routine, keep track of symbols
    // in case an infinite recursion is found (a symbol's value depending on itself).
    static std::set<std::string> recursive_symbols;

    Parser parser(parse_function);
    parser.registerVariables(varnames);

    std::set<std::string> symbols = parser.symbols();
    for (auto const& v : varnames) symbols.erase(v.c_str());

    // User can provide inputs under this name, through which expressions
    // can be provided for arbitrary variables. PICMI inputs are aware of
    // this convention and use the same prefix as well. This potentially
    // includes variable names that match physical or mathematical
    // constants, in case the user wishes to enforce a different
    // system of units or some form of quasi-physical behavior in the
    // simulation. Thus, this needs to override any built-in
    // constants.
    amrex::ParmParse pp_my_constants("my_constants");

    // Physical / Numerical Constants available to parsed expressions
    static std::map<std::string, amrex::Real> warpx_constants =
      {
      };

    for (auto it = symbols.begin(); it != symbols.end(); ) {
        // Always parsing in double precision avoids potential overflows that may occur when parsing
        // user's expressions because of the limited range of exponentials in single precision
        double v;

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(recursive_symbols.count(*it)==0, "Expressions contains recursive symbol "+*it);
        recursive_symbols.insert(*it);
        const bool is_input = queryWithParser(pp_my_constants, it->c_str(), v);
        recursive_symbols.erase(*it);

        if (is_input) {
            parser.setConstant(*it, v);
            it = symbols.erase(it);
            continue;
        }

        auto constant = warpx_constants.find(*it);
        if (constant != warpx_constants.end()) {
          parser.setConstant(*it, constant->second);
          it = symbols.erase(it);
          continue;
        }

        ++it;
    }
    for (auto const& s : symbols) {
        amrex::Abort("makeParser::Unknown symbol "+s);
    }
    return parser;
}

int safeCastToInt(const amrex::Real x, const std::string& real_name) {
    int result = 0;
    bool error_detected = false;
    std::string assert_msg;
    // (2.0*(numeric_limits<int>::max()/2+1)) converts numeric_limits<int>::max()+1 to a real ensuring accuracy to all digits
    // This accepts x = 2**31-1 but rejects 2**31.
    using namespace amrex::literals;
    constexpr amrex::Real max_range = (2.0_rt*static_cast<amrex::Real>(std::numeric_limits<int>::max()/2+1));
    if (x < max_range) {
        if (std::ceil(x) >= std::numeric_limits<int>::min()) {
            result = static_cast<int>(x);
        } else {
            error_detected = true;
            assert_msg = "Error: Negative overflow detected when casting " + real_name + " = " + std::to_string(x) + " to int";
        }
    } else if (x > 0) {
        error_detected = true;
        assert_msg =  "Error: Overflow detected when casting " + real_name + " = " + std::to_string(x) + " to int";
    } else {
        error_detected = true;
        assert_msg =  "Error: NaN detected when casting " + real_name + " to int";
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!error_detected, assert_msg);
    return result;
}

int
parseStringtoInt(std::string str, std::string name)
{
    auto const rval = static_cast<amrex::Real>(parseStringtoReal(str));
    int ival = safeCastToInt(std::round(rval), name);
    return ival;
}


namespace WarpXUtilMsg{

void AlwaysAssert(bool is_expression_true, const std::string& msg = "ERROR!")
{
    if(is_expression_true) {
        return;
    }
    amrex::Abort(msg);
}
} // namespace WarpXUtilMsg

namespace WarpXUtilStr
{
    auto is_in(const std::vector<std::string>& vect,
               const std::string& elem) -> bool
    {
        return (std::find(vect.begin(), vect.end(), elem) != vect.end());
    }

    auto is_in(const std::vector<std::string>& vect,
               const std::vector<std::string>& elems) -> bool
    {
        return std::any_of(elems.begin(), elems.end(),
            [&](const auto elem){return is_in(vect, elem);});
    }

    /** \brief Splits a string using a string separator. This is somewhat similar to
     *  amrex::Tokenize. The main difference is that, if the separator ":" is used,
     *  amrex::Tokenize will split ":3::2" into ["3","2"] while this functio will
     *  split ":3::2" into ["","3","","2"]. This function can also perform a trimming to
     *  remove whitespaces (or any other arbitrary string) from the split string.
     *
     * @tparam Container the type of the split string.
     *
     * @param[in] instr the input string
     * @param[in] separator the separator string
     * @param[in] trim true to trim the split string, false otherwise.
     * @param[in] trim_space the string to trim if trim is true.
     * @return cont the split string
     */
    template <typename Container>
    auto split (std::string const& instr, std::string const& separator,
                  bool const trim = false, std::string const& trim_space = " \t")
    {
        Container cont;
        std::size_t current = instr.find(separator);
        std::size_t previous = 0;
        while (current != std::string::npos) {
            if (trim){
                cont.push_back(amrex::trim(instr.substr(previous, current - previous),trim_space));}
            else{
                cont.push_back(instr.substr(previous, current - previous));}
            previous = current + separator.size();
            current = instr.find(separator, previous);
        }
        if (trim){
            cont.push_back(amrex::trim(instr.substr(previous, current - previous),trim_space));}
        else{
            cont.push_back(instr.substr(previous, current - previous));}
        return cont;
    }

} // namespace WarpXUtilStr



SliceParser::SliceParser (const std::string& instr)
{
    const std::string assert_msg = "ERROR: '" + instr + "' is not a valid syntax for a slice.";

    // split string and trim whitespaces
    auto insplit = WarpXUtilStr::split<std::vector<std::string>>(instr, m_separator, true);

    if(insplit.size() == 1){ // no colon in input string. The input is the period.
        m_period = parseStringtoInt(insplit[0], "interval period");}
    else if(insplit.size() == 2) // 1 colon in input string. The input is start:stop
    {
        if (!insplit[0].empty()){
            m_start = parseStringtoInt(insplit[0], "interval start");}
        if (!insplit[1].empty()){
            m_stop = parseStringtoInt(insplit[1], "interval stop");}
    }
    else // 2 colons in input string. The input is start:stop:period
    {
        WarpXUtilMsg::AlwaysAssert(insplit.size() == 3,assert_msg);
        if (!insplit[0].empty()){
            m_start = parseStringtoInt(insplit[0], "interval start");}
        if (!insplit[1].empty()){
            m_stop = parseStringtoInt(insplit[1], "interval stop");}
        if (!insplit[2].empty()){
            m_period = parseStringtoInt(insplit[2], "interval period");}
    }
}

bool SliceParser::contains (const int n) const
{
    if (m_period <= 0) {return false;}
    return (n - m_start) % m_period == 0 && n >= m_start && n <= m_stop;
}

int SliceParser::nextContains (const int n) const
{
    if (m_period <= 0) {return std::numeric_limits<int>::max();}
    int next = m_start;
    if (n >= m_start) {next = ((n-m_start)/m_period + 1)*m_period+m_start;}
    if (next > m_stop) {next = std::numeric_limits<int>::max();}
    return next;
}

int SliceParser::previousContains (const int n) const
{
    if (m_period <= 0) {return false;}
    int previous = ((std::min(n-1,m_stop)-m_start)/m_period)*m_period+m_start;
    if ((n < m_start) || (previous < 0)) {previous = 0;}
    return previous;
}

int SliceParser::getPeriod () const {return m_period;}

int SliceParser::getStart () const {return m_start;}

int SliceParser::getStop () const {return m_stop;}

IntervalsParser::IntervalsParser (const std::vector<std::string>& instr_vec)
{
    std::string inconcatenated;
    for (const auto& instr_element : instr_vec) inconcatenated +=instr_element;

    auto insplit = WarpXUtilStr::split<std::vector<std::string>>(inconcatenated, m_separator);

    for(const auto& inslc : insplit)
    {
        SliceParser temp_slice(inslc);
        m_slices.push_back(temp_slice);
        if ((temp_slice.getPeriod() > 0) &&
               (temp_slice.getStop() >= temp_slice.getStart())) m_activated = true;
    }
}

bool IntervalsParser::contains (const int n) const
{
    return std::any_of(m_slices.begin(), m_slices.end(),
        [&](const auto& slice){return slice.contains(n);});
}

int IntervalsParser::nextContains (const int n) const
{
    int next = std::numeric_limits<int>::max();
    for(const auto& slice: m_slices){
        next = std::min(slice.nextContains(n),next);
    }
    return next;
}

int IntervalsParser::previousContains (const int n) const
{
    int previous = 0;
    for(const auto& slice: m_slices){
        previous = std::max(slice.previousContains(n),previous);
    }
    return previous;
}

int IntervalsParser::previousContainsInclusive (const int n) const
{
    if (contains(n)){return n;}
    else {return previousContains(n);}
}

int IntervalsParser::localPeriod (const int n) const
{
    return nextContains(n) - previousContainsInclusive(n);
}

bool IntervalsParser::isActivated () const {return m_activated;}
