#ifndef INNER_OUTER_UPDATES_HPP_
#define INNER_OUTER_UPDATES_HPP_

namespace quokka {

auto innerUpdateRange(amrex::Box const &validBox, const int nghost)
    -> amrex::Box {
  // return interior box for this validBox
  return amrex::grow(validBox, -nghost);
}

auto outerUpdateRanges(amrex::Box const &validBox, const int nghost)
    -> std::vector<amrex::Box> {
  // return vector of outer boxes for this validBox
  std::vector<amrex::Box> boxes{};

  for (int i = 0; i < 2 * AMREX_SPACEDIM; ++i) {
    //amrex::Box computeRange = amrex::grow(validBox, nghost); // this is wrong
    amrex::Box computeRange = validBox;

    switch (i) { // FIXME: corners are not correctly included here
    case 0:
      computeRange.growHi(0, -(computeRange.length(0) - nghost)); // OK
      break;
    case 1:
      computeRange.growLo(0, -(computeRange.length(0) - nghost)); // OK
      break;
    case 2:
      computeRange.growHi(1, -(computeRange.length(1) - nghost)); // ?
      computeRange.grow(0, -nghost);
      break;
    case 3:
      computeRange.growLo(1, -(computeRange.length(1) - nghost)); // ?
      computeRange.grow(0, -nghost);
      break;
    case 4:
      computeRange.growHi(2, -(computeRange.length(2) - nghost)); // ?
      computeRange.grow(1, -nghost);
      computeRange.grow(0, -nghost);
      break;
    case 5:
      computeRange.growLo(2, -(computeRange.length(2) - nghost)); // ?
      computeRange.grow(1, -nghost);
      computeRange.grow(0, -nghost);
      break;
    }

    boxes.push_back(computeRange);
  }

  return boxes;
}

} // namespace quokka

#endif // INNER_OUTER_UPDATES_HPP_
