// derivative workspace -- grouped + self-sizing (see d.bind/d.count)
struct smokemax_constraint_derivs_t {
    double *chi_x;
    double *chi_y;
    double *chi_z;
    double *alpha_xx;
    double *alpha_yy;
    double *alpha_zz;
    double *chi_xx;
    double *chi_yy;
    double *chi_zz;
    double *trK_xx;
    double *trK_yy;
    double *trK_zz;
    static constexpr unsigned int count() { return 12; }
    void bind(double *deriv_base, unsigned int BLK_SZ) {
        chi_x = deriv_base + 0 * BLK_SZ;
        chi_y = deriv_base + 1 * BLK_SZ;
        chi_z = deriv_base + 2 * BLK_SZ;
        alpha_xx = deriv_base + 3 * BLK_SZ;
        alpha_yy = deriv_base + 4 * BLK_SZ;
        alpha_zz = deriv_base + 5 * BLK_SZ;
        chi_xx = deriv_base + 6 * BLK_SZ;
        chi_yy = deriv_base + 7 * BLK_SZ;
        chi_zz = deriv_base + 8 * BLK_SZ;
        trK_xx = deriv_base + 9 * BLK_SZ;
        trK_yy = deriv_base + 10 * BLK_SZ;
        trK_zz = deriv_base + 11 * BLK_SZ;
    }
};
