// derivative workspace -- grouped + self-sizing (see d.bind/d.count)
struct smokemin_evolution_derivs_t {
    double *u_x;
    double *u_y;
    double *u_z;
    double *v_x;
    double *v_y;
    double *v_z;
    double *u_xx;
    double *u_yy;
    double *u_zz;
    static constexpr unsigned int count() { return 9; }
    void bind(double *deriv_base, unsigned int BLK_SZ) {
        u_x = deriv_base + 0 * BLK_SZ;
        u_y = deriv_base + 1 * BLK_SZ;
        u_z = deriv_base + 2 * BLK_SZ;
        v_x = deriv_base + 3 * BLK_SZ;
        v_y = deriv_base + 4 * BLK_SZ;
        v_z = deriv_base + 5 * BLK_SZ;
        u_xx = deriv_base + 6 * BLK_SZ;
        u_yy = deriv_base + 7 * BLK_SZ;
        u_zz = deriv_base + 8 * BLK_SZ;
    }
};
