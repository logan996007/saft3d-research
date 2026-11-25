// Float64 geometry batches for the SAFT-inspired research prototype.
// Compiled at runtime by CuPy/NVRTC. No CPU compiler or external headers needed.
// SAT tolerances are absolute projected lengths after axis normalization.

__device__ double dot3(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ void sub3(const double* a, const double* b, double* r) {
    for (int k=0; k<3; ++k) r[k] = a[k] - b[k];
}

__device__ void cross3(const double* a, const double* b, double* r) {
    r[0] = a[1]*b[2] - a[2]*b[1];
    r[1] = a[2]*b[0] - a[0]*b[2];
    r[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ double six_volume(const double* t) {
    double a[3], b[3], c[3], normal[3];
    sub3(t+3, t, a); sub3(t+6, t, b); sub3(t+9, t, c);
    cross3(a, b, normal);
    return dot3(normal, c);
}

__device__ void bounds(const double* t, double* lo, double* hi) {
    for (int k=0; k<3; ++k) {
        lo[k] = hi[k] = t[k];
        for (int v=1; v<4; ++v) {
            lo[k] = fmin(lo[k], t[3*v+k]);
            hi[k] = fmax(hi[k], t[3*v+k]);
        }
    }
}

// A zero cross product represents parallel edges and adds no separating axis.
__device__ bool separated(const double* a, const double* b,
                          const double* axis, const double tol) {
    const double norm = sqrt(dot3(axis, axis));
    if (norm == 0.) return false;
    double n[3] = {axis[0]/norm, axis[1]/norm, axis[2]/norm};
    // Subtract one common origin to reduce cancellation in projections.
    double alo=0., ahi=0., blo=0., bhi=0.;
    for (int v=0; v<4; ++v) {
        double av[3], bv[3];
        sub3(a+3*v, a, av); sub3(b+3*v, a, bv);
        const double pa=dot3(av,n), pb=dot3(bv,n);
        if (v==0) { alo=ahi=pa; blo=bhi=pb; }
        else {
            alo=fmin(alo,pa); ahi=fmax(ahi,pa);
            blo=fmin(blo,pb); bhi=fmax(bhi,pb);
        }
    }
    return fmin(ahi,bhi) - fmax(alo,blo) <= tol;
}

__device__ bool positive_overlap(const double* a, const double* b,
                                 const double tol) {
    double alo[3], ahi[3], blo[3], bhi[3];
    bounds(a,alo,ahi); bounds(b,blo,bhi);
    for (int k=0; k<3; ++k)
        if (fmin(ahi[k],bhi[k])-fmax(alo[k],blo[k]) <= tol) return false;
    // A geometrically flat tetrahedron has no volume to intersect.
    if (six_volume(a)==0. || six_volume(b)==0.) return false;
    const int faces[4][3] = {{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
    const int edges[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for (int side=0; side<2; ++side) {
        const double* t = side==0 ? a : b;
        for (int f=0; f<4; ++f) {
            double u[3], v[3], axis[3];
            sub3(t+3*faces[f][1],t+3*faces[f][0],u);
            sub3(t+3*faces[f][2],t+3*faces[f][0],v);
            cross3(u,v,axis);
            if (separated(a,b,axis,tol)) return false;
        }
    }
    for (int i=0; i<6; ++i) {
        double u[3];
        sub3(a+3*edges[i][1],a+3*edges[i][0],u);
        for (int j=0; j<6; ++j) {
            double v[3], axis[3];
            sub3(b+3*edges[j][1],b+3*edges[j][0],v);
            cross3(u,v,axis);
            // Match the CPU predicate's relative near-parallel cutoff. Skipping
            // this axis is conservative: it can add a conflict, not remove one.
            const double cutoff=7.105427357601002e-15*sqrt(dot3(u,u))*sqrt(dot3(v,v));
            if (sqrt(dot3(axis,axis))<=cutoff) continue;
            if (separated(a,b,axis,tol)) return false;
        }
    }
    return true;
}

// Preserve the shared visibility-kernel anchor outside/on each filled-tet side.
__device__ bool visible_anchor(const double* tet, const double* anchor, double tol) {
    const int sides[3][4]={{0,1,3,2},{0,3,2,1},{1,2,3,0}};
    for (int s=0; s<3; ++s) {
        const double* x=tet+3*sides[s][0];
        double u[3],v[3],other[3],to_anchor[3],normal[3];
        sub3(tet+3*sides[s][1],x,u); sub3(tet+3*sides[s][2],x,v);
        sub3(tet+3*sides[s][3],x,other); sub3(anchor,x,to_anchor);
        cross3(u,v,normal);
        const double norm=sqrt(dot3(normal,normal));
        if (norm==0.) return false;
        const double sign=dot3(normal,other)>0. ? -1. : 1.;
        if (sign*dot3(normal,to_anchor)/norm < -tol) return false;
    }
    return true;
}

extern "C" __global__ void propose_tets(const double* triangles, const long long n,
                                       const double* anchor, const double target_size,
                                       const unsigned char* close_flags, const double tol,
                                       double* candidates, unsigned char* valid) {
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const double* tri=triangles+9*i;
    double u[3],v[3],normal[3],center[3],to_anchor[3],base_to_anchor[3];
    sub3(tri+3,tri,u); sub3(tri+6,tri,v); cross3(u,v,normal);
    const double normal_len=sqrt(dot3(normal,normal));
    for (int k=0; k<3; ++k) {
        normal[k]=normal_len>0. ? normal[k]/normal_len : 0.;
        center[k]=(tri[k]+tri[3+k]+tri[6+k])/3.;
    }
    sub3(anchor,center,to_anchor); sub3(anchor,tri,base_to_anchor);
    const double depth=-dot3(normal,base_to_anchor);
    const double ideal_h=fmin(sqrt(2./3.)*target_size,.65*depth);
    const double factors[3]={1.,.75,.5};
    bool any_growth=false;
    for (int slot=0; slot<6; ++slot) {
        double* tet=candidates+(i*6+slot)*12;
        for (int k=0; k<3; ++k) {
            tet[k]=tri[k]; tet[3+k]=tri[6+k]; tet[6+k]=tri[3+k];
            if (slot<3) tet[9+k]=center[k]-normal[k]*ideal_h*factors[slot];
            else if (slot==3) tet[9+k]=center[k]+to_anchor[k]*(depth>0. ? fmin(ideal_h/depth,.65) : 0.);
            else if (slot==4) tet[9+k]=center[k]+to_anchor[k]*.35;
            else tet[9+k]=anchor[k];
        }
        bool ok=six_volume(tet)/6.>tol*tol;
        if (slot<5) {
            ok=ok && !close_flags[i] && normal_len>0. && depth>0. && visible_anchor(tet,anchor,tol);
            any_growth=any_growth || ok;
        } else ok=ok && (close_flags[i] || !any_growth);
        valid[i*6+slot]=ok ? 1 : 0;
    }
}

// General advancing-front proposals: local inward normal, no kernel anchor.
// Intersection, containment and topological acceptance are separate stages.
extern "C" __global__ void propose_local_tets(const double* triangles, const long long n,
                                             const double* target_sizes, const double tol,
                                             double* candidates, unsigned char* valid) {
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const double* tri=triangles+9*i;
    double u[3],v[3],w[3],normal[3],center[3];
    sub3(tri+3,tri,u); sub3(tri+6,tri,v); sub3(tri+6,tri+3,w);
    cross3(u,v,normal);
    const double normal_len=sqrt(dot3(normal,normal));
    const double mean_edge=(sqrt(dot3(u,u))+sqrt(dot3(v,v))+sqrt(dot3(w,w)))/3.;
    const double height=sqrt(2./3.)*fmin(target_sizes[i],mean_edge);
    const double factors[6]={1.,.8,.6,.4,.2,.1};
    for (int k=0; k<3; ++k) {
        normal[k]=normal_len>0. ? normal[k]/normal_len : 0.;
        center[k]=(tri[k]+tri[3+k]+tri[6+k])/3.;
    }
    for (int slot=0; slot<6; ++slot) {
        double* tet=candidates+(i*6+slot)*12;
        for (int k=0; k<3; ++k) {
            tet[k]=tri[k]; tet[3+k]=tri[6+k]; tet[6+k]=tri[3+k];
            tet[9+k]=center[k]-normal[k]*height*factors[slot];
        }
        valid[i*6+slot]=(six_volume(tet)/6.>tol*tol) ? 1 : 0;
    }
}

extern "C" __global__ void evaluate_tets(const double* tets, const long long n,
                                        double* volumes, double* quality) {
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const double* t=tets+12*i;
    const double volume=six_volume(t)/6.;
    double sum=0.;
    for (int a=0; a<4; ++a)
        for (int b=a+1; b<4; ++b)
            for (int k=0; k<3; ++k) {
                const double d=t[3*a+k]-t[3*b+k]; sum+=d*d;
            }
    volumes[i]=volume;
    quality[i]=(volume>0. && sum>0.) ? 12.*pow(3.*volume,2./3.)/sum : 0.;
}

extern "C" __global__ void aabb_conflicts(const double* tets, const long long n,
                                          const double padding, unsigned char* out) {
    const long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=n*n) return;
    const long long i=idx/n, j=idx%n;
    if (i==j) { out[idx]=0; return; }
    double alo[3], ahi[3], blo[3], bhi[3];
    bounds(tets+12*i,alo,ahi); bounds(tets+12*j,blo,bhi);
    for (int k=0; k<3; ++k)
        if (alo[k]-padding>bhi[k]+padding || blo[k]-padding>ahi[k]+padding) {
            out[idx]=0; return;
        }
    out[idx]=1;
}

extern "C" __global__ void tet_overlaps(const double* candidates, const long long n,
                                       const double* existing, const long long m,
                                       const double tol, unsigned char* out) {
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    out[i]=0;
    for (long long j=0; j<m; ++j)
        if (positive_overlap(candidates+12*i,existing+12*j,tol)) {
            out[i]=1; return;
        }
}
