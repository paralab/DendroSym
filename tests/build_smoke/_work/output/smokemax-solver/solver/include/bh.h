/**
 * @file bh.h
 * @brief BH struct for the smokemax solver.
 *
 * Wraps the generic dendro_gr::BH struct, adding a project-specific
 * namespace alias. If your theory needs extra BH fields (e.g., charge),
 * extend this struct here.
 */

#ifndef SMOKEMAX_BH_H
#define SMOKEMAX_BH_H

// use the generic BH struct from dendrolib GR/
#include "dendro_gr_bh.h"

namespace smokemax {
    // alias the generic BH struct into our namespace
    using BH = dendro_gr::BH;
}  // namespace smokemax

#endif  // SMOKEMAX_BH_H
